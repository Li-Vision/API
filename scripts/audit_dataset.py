"""
Audita datasets JA' gravados: acha amostras corrompidas ou fora do padrao.

A validacao na coleta (src/services/collection_queue.py) protege o que entra
daqui para frente; este script passa a mesma regua no que ja' esta' em
`samples` — coletado antes da validacao existir, ou antes de o sinal ter
amostras suficientes para comparar.

Para cada amostra:
  rejected  -> corrompida (NaN, tamanho errado, mao colapsada, sequencia
               congelada...). Nao ha' o que aprender com ela.
  suspect   -> integra, mas destoa do proprio sinal ou parece outro sinal.
  accepted  -> ok.

Uso (da pasta API):

    python scripts/audit_dataset.py --list
    python scripts/audit_dataset.py --datasets ALFABETO_HOLISTICO
    python scripts/audit_dataset.py --all --report auditoria.json
    python scripts/audit_dataset.py --datasets LIBRAS --with-model
    python scripts/audit_dataset.py --datasets LIBRAS --apply      # move p/ quarentena

Sem --apply o script SO' LE. Com --apply, rejected e suspect saem de
`samples` e vao para `samples_quarantine` (com o motivo); nada e' apagado de
vez, e o admin pode devolver pelo endpoint /collect/quarantine/{id}/approve.
"""
import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.validation import sample_validator as sv  # noqa: E402


def audit_rows(rows, kind: str, cfg: sv.ValidationConfig, model_check=None):
    """
    Veredito de cada linha {"id", "label", "features", ...} contra o resto do
    dataset. Logica pura — testada em tests/test_sample_validator.py.
    """
    ref = sv.DatasetReference.from_rows(rows, kind, cfg=cfg)
    results = []
    for row in rows:
        label = str(row["label"])
        verdict = sv.validate(row["features"], label, kind, reference=ref,
                              model_check=model_check, cfg=cfg)
        results.append({
            "id": row.get("id"),
            "label": label,
            "user_id": row.get("user_id"),
            "created_at": row.get("created_at"),
            **verdict.to_dict(),
        })
    return results


def summarize(results):
    per_label = defaultdict(Counter)
    for r in results:
        per_label[r["label"]][r["status"]] += 1
    return per_label


def _worst_z(r) -> float:
    return (r.get("scores", {}).get("similarity") or {}).get("z_own") or 0.0


def print_report(name: str, kind: str, results) -> None:
    per_label = summarize(results)
    total = Counter(r["status"] for r in results)
    print(f"\n=== {name} ({kind}) — {len(results)} amostras ===")
    print(f"  aceitas {total['accepted']} | suspeitas {total['suspect']} | rejeitadas {total['rejected']}")
    print(f"  {'sinal':20} {'total':>6} {'ok':>6} {'susp':>6} {'rej':>6}")
    for label in sorted(per_label):
        c = per_label[label]
        n = sum(c.values())
        flag = "  <-- " if (c["suspect"] + c["rejected"]) / n > 0.1 else ""
        print(f"  {label:20} {n:6} {c['accepted']:6} {c['suspect']:6} {c['rejected']:6}{flag}")

    bad = [r for r in results if r["status"] != sv.STATUS_ACCEPTED]
    bad.sort(key=lambda r: (r["status"] != sv.STATUS_REJECTED, -_worst_z(r)))
    if bad:
        print("  Piores casos:")
        for r in bad[:15]:
            why = "; ".join(x["message"] for x in r["reasons"] if x["severity"] != "info")
            print(f"    [{r['status']:8}] {r['label']:12} id={r['id']}  {why}")
        if len(bad) > 15:
            print(f"    ... e mais {len(bad) - 15} (use --report para a lista completa)")


def move_to_quarantine(supabase, dataset_id: str, kind: str, rows_by_id: dict, results) -> int:
    from src.services.collection_queue import QUARANTINE_TABLE

    moved = 0
    for r in results:
        if r["status"] == sv.STATUS_ACCEPTED or r["id"] is None:
            continue
        row = rows_by_id[r["id"]]
        item = {
            "dataset_id": dataset_id,
            "label": r["label"],
            "kind": kind,
            "features": row["features"],
            "reasons": r["reasons"],
            "scores": {**r["scores"], "audit_status": r["status"], "original_sample_id": r["id"]},
        }
        if row.get("user_id"):
            item["user_id"] = row["user_id"]
        # Copia antes de apagar: se o delete falhar, sobra duplicata, nunca perda.
        supabase.table(QUARANTINE_TABLE).insert(item).execute()
        supabase.table("samples").delete().eq("id", r["id"]).execute()
        moved += 1
    return moved


def main() -> None:
    parser = argparse.ArgumentParser(description="Audita amostras gravadas nos datasets.")
    parser.add_argument("--list", action="store_true", help="lista os datasets e sai")
    parser.add_argument("--datasets", nargs="+", help="nomes ou ids dos datasets")
    parser.add_argument("--all", action="store_true", help="audita todos os datasets")
    parser.add_argument("--report", help="grava o resultado completo neste arquivo JSON")
    parser.add_argument("--with-model", action="store_true",
                        help="compara tambem com os modelos ativos (pastas do config.yaml)")
    parser.add_argument("--suspect-z", type=float, help="limiar de 'fora do padrao' (padrao 4.0)")
    parser.add_argument("--apply", action="store_true",
                        help="move rejeitadas e suspeitas de samples para samples_quarantine")
    parser.add_argument("--yes", action="store_true", help="nao pede confirmacao no --apply")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s [%(name)s] %(message)s")

    from src.core.supabase_client import supabase
    from src.core.pagination import fetch_all

    datasets = sorted(supabase.table("datasets").select("id, name, type").execute().data,
                      key=lambda r: r["name"])
    if args.list:
        for d in datasets:
            print(f'{d["type"]:8} {d["name"]:40} {d["id"]}')
        return

    if args.all:
        chosen = datasets
    elif args.datasets:
        by_id = {d["id"]: d for d in datasets}
        by_name = {d["name"].casefold(): d for d in datasets}
        chosen = []
        for ref in args.datasets:
            d = by_id.get(ref) or by_name.get(ref.casefold())
            if not d:
                sys.exit(f"Dataset nao encontrado: {ref!r}. Use --list.")
            chosen.append(d)
    else:
        parser.error("informe --datasets, --all ou --list")

    cfg = sv.ValidationConfig()
    if args.suspect_z:
        cfg.suspect_z = args.suspect_z

    if args.with_model:
        from src.core.config_loader import Config
        from src.core.model_cache import ModelCache
        ModelCache.initialize(Config(str(ROOT / "config.yaml")))

    report = {}
    for d in chosen:
        kind = sv.KIND_DYNAMIC if d["type"] == "dynamic" else sv.KIND_STATIC
        page = 100 if kind == sv.KIND_DYNAMIC else 1000
        rows = fetch_all(
            lambda: supabase.table("samples").select("id, label, features, user_id, created_at")
            .eq("dataset_id", d["id"]),
            page_size=page,
        )
        if not rows:
            print(f"\n=== {d['name']}: vazio ===")
            continue

        model_check = None
        if args.with_model:
            from src.validation.model_check import build_model_check
            names = [m["name"] for m in supabase.table("models").select("name")
                     .eq("dataset_id", d["id"]).execute().data]
            model_check = build_model_check(kind, names)

        results = audit_rows(rows, kind, cfg, model_check)
        print_report(d["name"], kind, results)
        report[d["name"]] = {"dataset_id": d["id"], "kind": kind, "results": results}

        bad = sum(r["status"] != sv.STATUS_ACCEPTED for r in results)
        if args.apply and bad:
            if not args.yes:
                answer = input(f"Mover {bad} amostra(s) de {d['name']} para a quarentena? (s/N): ")
                if answer.strip().lower() not in ("s", "sim", "y"):
                    print("  Pulado.")
                    continue
            moved = move_to_quarantine(supabase, d["id"], kind, {r["id"]: r for r in rows}, results)
            print(f"  {moved} amostra(s) movida(s) para samples_quarantine.")

    if args.report:
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str),
                                     encoding="utf-8")
        print(f"\nRelatorio completo: {args.report}")


if __name__ == "__main__":
    main()
