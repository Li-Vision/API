"""
Treina um modelo NESTA maquina e publica o resultado no mesmo Supabase do servidor.

Por que existe: o treino no Render compete por memoria com a API (torch +
mediapipe + sklearn ja ocupam ~300-470 MB dos 512 MB). Aqui o treino usa a
RAM do computador e o servidor so' baixa o arquivo pronto quando o modelo e'
ativado.

O fluxo e' o mesmo do servidor (TrainingService.train_model): le as amostras
do Supabase, treina, sobe o arquivo para o bucket `models`, atualiza as
tabelas `models` e `training_jobs`. Depois e' so' ativar o modelo pelo app.

Uso (da pasta API, com o ambiente de treino ativo — ver requirements-train.txt):

    python scripts/train_local.py            # abre o menu interativo (ou treinar.bat)
    python scripts/train_local.py --list
    python scripts/train_local.py --model ALFABETO --datasets ALFABETO_HOLÍSTICO
    python scripts/train_local.py --model LIBRAS --datasets <id-ou-nome> <id-ou-nome>
"""
import argparse
import logging
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def pinned_sklearn_version() -> str | None:
    """Versao de scikit-learn fixada no requirements.txt da API (a do servidor)."""
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*scikit-learn\s*==\s*([\w.]+)", line)
        if m:
            return m.group(1)
    return None


def check_sklearn(force: bool) -> None:
    """
    O .joblib estatico e' um pickle do sklearn: gerado numa versao diferente da
    do servidor ele pode nao carregar (ou carregar com comportamento errado).
    """
    import sklearn

    expected = pinned_sklearn_version()
    if expected and sklearn.__version__ != expected:
        msg = (
            f"scikit-learn local = {sklearn.__version__}, servidor = {expected}. "
            "Um modelo treinado assim pode nao carregar no servidor. "
            "Use o ambiente de treino (pip install -r requirements-train.txt)."
        )
        if not force:
            sys.exit("ERRO: " + msg + " Use --force para ignorar.")
        print("AVISO: " + msg)


def resolve_dataset_ids(supabase, refs: list[str]) -> list[str]:
    rows = supabase.table("datasets").select("id, name, type").execute().data
    by_id = {r["id"]: r for r in rows}
    by_name = {r["name"].casefold(): r for r in rows}
    ids = []
    for ref in refs:
        row = by_id.get(ref) or by_name.get(ref.casefold())
        if not row:
            sys.exit(f"Dataset nao encontrado: {ref!r}. Use --list para ver os disponiveis.")
        ids.append(row["id"])
    return ids


def list_datasets(supabase) -> list[dict]:
    return sorted(supabase.table("datasets").select("id, name, type").execute().data, key=lambda r: r["name"])


def run_training(ids: list[str], model: str) -> bool:
    """Treina e publica. Retorna True se todos os tipos (static/dynamic) terminaram bem."""
    from src.services.training_service import TrainingService

    print(f"Treinando {model.upper()} com {len(ids)} dataset(s) — o resultado sera publicado no Supabase.")
    result = TrainingService().train_model(ids, model)

    failed = [d for d in result.get("details", []) if d.get("status") == "failed"]
    for d in result.get("details", []):
        status = d.get("status")
        extra = f'acuracia={d["accuracy"]:.3f}' if d.get("accuracy") is not None else d.get("error", "")
        print(f'  [{d.get("type", "?")}] {status} {extra}')
    if result.get("status") == "failed" or failed:
        return False
    print("Pronto. Ative o modelo pelo app (ou POST /train/activate) para o servidor baixa-lo.")
    return True


def print_datasets(rows: list[dict]) -> None:
    for i, r in enumerate(rows, 1):
        print(f'  {i:2}) {r["type"]:8} {r["name"]}')


def parse_selection(text: str, size: int) -> list[int] | None:
    """'1,3 5-7' -> [0, 2, 4, 5, 6]. None se algo for invalido."""
    picked: list[int] = []
    for part in text.replace(",", " ").split():
        lo, _, hi = part.partition("-")
        if not lo.isdigit() or (hi and not hi.isdigit()):
            return None
        a, b = int(lo), int(hi or lo)
        if a < 1 or b > size or a > b:
            return None
        picked += [i - 1 for i in range(a, b + 1) if i - 1 not in picked]
    return picked or None


def menu(supabase, force: bool) -> None:
    while True:
        print("\n=== Li-Vision :: treino local ===")
        print("  1) Listar datasets")
        print("  2) Treinar modelo")
        print("  3) Listar modelos publicados")
        print("  0) Sair")
        choice = input("Opcao: ").strip()

        if choice == "0":
            return
        if choice == "1":
            print_datasets(list_datasets(supabase))
        elif choice == "3":
            for m in supabase.table("models").select("name, type, accuracy, total_samples_trained, updated_at").order("name").execute().data:
                acc = f'{m["accuracy"]:.3f}' if m.get("accuracy") is not None else "-"
                print(f'  {m["name"]:30} {m["type"]:8} acc={acc} amostras={m.get("total_samples_trained")} {m.get("updated_at", "")[:19]}')
        elif choice == "2":
            rows = list_datasets(supabase)
            if not rows:
                print("Nenhum dataset encontrado.")
                continue

            # Modelo: reaproveita um existente (o treino atualiza a mesma linha em
            # `models`, sem criar outro) ou cria um novo pelo nome.
            models = supabase.table("models").select("name, type, dataset_id").execute().data
            names_existing = sorted({m["name"] for m in models})
            print("Modelo:")
            for i, n in enumerate(names_existing, 1):
                print(f"  {i:2}) {n}")
            print("   N) novo modelo")
            pick = input("Retreinar qual modelo? (numero, N para novo, vazio cancela): ").strip()
            if not pick:
                print("Cancelado.")
                continue
            if pick.lower() == "n":
                model = input("Nome do novo modelo (ex.: ALFABETO): ").strip()
                suggested: list[int] = []
            elif pick.isdigit() and 1 <= int(pick) <= len(names_existing):
                model = names_existing[int(pick) - 1]
                # Cada linha de `models` guarda so' o 1o dataset do treino; serve de
                # sugestao (os que ainda existem) e voce pode acrescentar outros.
                known = {m["dataset_id"] for m in models if m["name"] == model and m.get("dataset_id")}
                suggested = [i for i, r in enumerate(rows) if r["id"] in known]
            else:
                print("Opcao invalida.")
                continue
            if not model:
                print("Cancelado.")
                continue

            print_datasets(rows)
            hint = ""
            if suggested:
                hint = f" [Enter = {', '.join(rows[i]['name'] for i in suggested)}]"
            answer = input(f"Datasets (numeros, ex.: 1,3 ou 2-4){hint}: ").strip()
            if not answer and suggested:
                sel = suggested
            else:
                sel = parse_selection(answer, len(rows))
            if sel is None:
                print("Selecao vazia ou invalida.")
                continue
            names = ", ".join(rows[i]["name"] for i in sel)
            if input(f"Treinar {model.upper()} com [{names}]? (s/N): ").strip().lower() not in ("s", "sim", "y"):
                print("Cancelado.")
                continue
            check_sklearn(force)
            try:
                run_training([rows[i]["id"] for i in sel], model)
            except Exception as exc:  # o menu continua aberto para tentar de novo
                print(f"ERRO no treino: {exc}")
        else:
            print("Opcao invalida.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina um modelo localmente e publica no Supabase.")
    parser.add_argument("--list", action="store_true", help="lista os datasets e sai (somente leitura)")
    parser.add_argument("--model", help="nome do modelo (ex.: ALFABETO)")
    parser.add_argument("--datasets", nargs="+", help="nomes ou ids dos datasets")
    parser.add_argument("--force", action="store_true", help="ignora a checagem de versao do scikit-learn")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    from src.core.supabase_client import supabase

    if args.list:
        for r in list_datasets(supabase):
            print(f'{r["type"]:8} {r["name"]:40} {r["id"]}')
        return

    if not args.model and not args.datasets:
        menu(supabase, args.force)
        return
    if not args.model or not args.datasets:
        parser.error("informe --model e --datasets (ou use --list, ou nenhum argumento para o menu)")

    check_sklearn(args.force)
    ids = resolve_dataset_ids(supabase, args.datasets)
    if not run_training(ids, args.model):
        sys.exit(1)


if __name__ == "__main__":
    main()
