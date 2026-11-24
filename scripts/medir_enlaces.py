"""
Mede o tamanho (bytes) e o tempo das mensagens de cada enlace do Li-Vision
(Secao 6 da Reposicao). Cada medicao e repetida N vezes (padrao 10) e sai com
minimo / media / maximo, em texto pronto para colar no documento.

Uso (na pasta API):
    python scripts/medir_enlaces.py ws                 # E1/E2  WebSocket (servidor real)
    python scripts/medir_enlaces.py e5                 # E5     corpo do INSERT em samples
    python scripts/medir_enlaces.py e6                 # E6     tamanho dos modelos no Storage (so leitura)
    LV_EMAIL=.. LV_PASSWORD=.. python scripts/medir_enlaces.py rest   # E3  rotas REST
    LV_EMAIL=.. LV_PASSWORD=.. python scripts/medir_enlaces.py auth   # E4  Supabase Auth
    python scripts/medir_enlaces.py all                # tudo que nao exige senha

Opcoes: --n 10  --api https://li-vision-api.onrender.com

O que e MEDIDO e o que e CONSTRUIDO (diga isto no documento):
  ws    bytes recebidos = medidos (resposta real do servidor). Bytes enviados =
        JSON montado aqui com a MESMA estrutura e formatacao numerica do app
        (holisticFeatures.ts: mao 21 pts + pose 33 + rosto 478 + aspect). Os
        valores dos pontos sao sinteticos; o tamanho varia pouco com eles.
  e5    corpo do INSERT montado com o codigo real build_frame_vector da API.
  e6    tamanhos lidos do Storage (listagem, sem baixar nada).
  rest/auth  medidos na rede (exigem login; senha so por variavel de ambiente).

Nao escreve nada em banco nem em storage. NAO usa start_collect.
Bytes de TLS/TCP/IP nao entram: sao a camada de baixo do enlace.
"""

import argparse
import asyncio
import json
import os
import statistics as st
import sys
import time
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_API = "https://li-vision-api.onrender.com"
SEP = (",", ":")  # JSON.stringify do JS e compacto


def credenciais():
    """Login de teste: LV_EMAIL/LV_PASSWORD, ou Email/Senha no .env da raiz do repo."""
    from dotenv import load_dotenv

    load_dotenv(ROOT.parent / ".env")
    return (os.getenv("LV_EMAIL") or os.getenv("Email"),
            os.getenv("LV_PASSWORD") or os.getenv("Senha"))


# ---------------------------------------------------------------- utilidades
RAW = []  # (id, enlace, medida, unidade, valores) -> medicoes_raw.csv


def stats(label, values, unit, rid=None, enlace=""):
    if rid and values:
        RAW.append((rid, enlace, label, unit, list(values)))
    if not values:
        print(f"  {label}: sem dados")
        return
    print(
        f"  {label:<34} n={len(values):<3} min={min(values):>10.1f}  "
        f"media={st.mean(values):>10.1f}  max={max(values):>10.1f} {unit}"
    )


def js_float(rng):
    """Numero como o app o envia: float32 do MediaPipe promovido a double.

    repr() do Python e o JSON.stringify do JS dao a mesma cadeia (menor
    representacao que volta ao mesmo double), entao o tamanho e fiel.
    """
    return float(np.float32(rng.random()))


def point(rng, vis=False):
    p = {"x": js_float(rng), "y": js_float(rng), "z": float(np.float32(-rng.random() * 0.1))}
    if vis:
        p["visibility"] = js_float(rng)
    return p


def payload_holistic(rng, hands=1):
    return {
        "hands": [[point(rng) for _ in range(21)] for _ in range(hands)],
        "pose": [point(rng, vis=True) for _ in range(33)],
        "face": [point(rng) for _ in range(478)],
        "aspect": 0.75,
    }


def payload_hands(rng, hands=1):
    return [[point(rng) for _ in range(21)] for _ in range(hands)]


def ws_overhead(n, masked):
    """Cabecalho do quadro WebSocket (RFC 6455) para n bytes de payload."""
    ext = 0 if n < 126 else (2 if n <= 0xFFFF else 8)
    return 2 + ext + (4 if masked else 0)


# --------------------------------------------------------------------- E1/E2
async def medir_ws(api, n):
    import websockets

    url = api.replace("http", "ws", 1) + "/ws/detect?mode=hybrid&use_rules=false"
    rng = np.random.default_rng(1)
    print(f"\n== E1/E2  WebSocket  {url}")
    t0 = time.perf_counter()
    try:
        async with websockets.connect(url, open_timeout=120, max_size=None) as ws:
            print(f"  conexao aberta em {time.perf_counter() - t0:.2f} s "
                  f"(>20 s indica cold start do Render)")
            for nome, fabrica in (("holistic_v1", payload_holistic), ("hands_v1", payload_hands)):
                up, down, rtt = [], [], []
                erros = 0
                for _ in range(n):
                    msg = json.dumps(fabrica(rng), separators=SEP)
                    b = msg.encode("utf-8")
                    t = time.perf_counter()
                    await ws.send(msg)
                    try:
                        resp = await asyncio.wait_for(ws.recv(), timeout=15)
                    except asyncio.TimeoutError:
                        erros += 1
                        continue
                    rtt.append((time.perf_counter() - t) * 1000)
                    rb = resp.encode("utf-8") if isinstance(resp, str) else resp
                    up.append(len(b))
                    down.append(len(rb))
                    if "error" in json.loads(rb):
                        erros += 1
                print(f"\n  esquema {nome} (1 mao)  respostas com erro/timeout: {erros}")
                stats("mensagem enviada (payload)", up, "B", f"E1_up_{nome}", "E1")
                stats("  + cabecalho WS (mascarado)", [x + ws_overhead(x, True) for x in up], "B")
                stats("resposta recebida (payload)", down, "B", f"E2_down_{nome}", "E2")
                stats("  + cabecalho WS", [x + ws_overhead(x, False) for x in down], "B")
                stats("RTT envio -> resposta", rtt, "ms", f"E1E2_rtt_{nome}", "E1/E2")
                if down:
                    print("  exemplo de resposta:", resp[:300] if isinstance(resp, str) else "<bytes>")
    except Exception as e:
        print(f"  FALHOU: {type(e).__name__}: {e}")


# ------------------------------------------------------------------------ E5
def medir_e5(n):
    from src.api.holistic import parse_input
    from src.data_collection import holistic_features as hf

    print("\n== E5  corpo do INSERT em samples (PostgREST)")
    rng = np.random.default_rng(2)
    ds, uid = str(uuid.uuid4()), str(uuid.uuid4())
    est, din = [], []
    for _ in range(n):
        frame = parse_input(payload_holistic(rng))
        vec = hf.build_frame_vector(frame, hf.SCHEMA_HOLISTIC_V1)
        est.append(len(json.dumps(
            {"dataset_id": ds, "label": "G", "features": vec, "user_id": uid}, separators=SEP)))
        flat = []
        for _ in range(15):
            flat.extend(hf.build_frame_vector(
                parse_input(payload_holistic(rng)), hf.SCHEMA_HOLISTIC_V1))
        din.append(len(json.dumps(
            {"dataset_id": ds, "label": "G", "features": flat, "user_id": uid}, separators=SEP)))
    print(f"  features por frame (holistic_v1): {hf.frame_size(hf.SCHEMA_HOLISTIC_V1)} valores")
    stats("coleta estatica (1 frame)", est, "B", "E5_insert_estatica", "E5")
    stats("coleta dinamica (15 frames)", din, "B", "E5_insert_dinamica", "E5")
    print("  obs.: cabecalhos HTTP do PostgREST nao incluidos (medir no painel do Supabase).")


# ------------------------------------------------------------------------ E6
def medir_e6():
    print("\n== E6  Supabase Storage / banco (somente leitura)")
    from src.core.supabase_client import supabase

    try:
        models = supabase.table("models").select("id,name,storage_path").execute().data or []
    except Exception as e:
        print(f"  FALHOU lendo tabela models: {e}")
        return
    sizes = []
    for m in models:
        path = m.get("storage_path")
        if not path:
            continue
        folder, _, name = path.rpartition("/")
        try:
            for it in supabase.storage.from_("models").list(folder):
                if it.get("name") == name:
                    sz = (it.get("metadata") or {}).get("size")
                    sizes.append(sz)
                    print(f"  {path:<60} {sz} B")
        except Exception as e:
            print(f"  {path}: falhou ({e})")
    sizes = [s for s in sizes if isinstance(s, (int, float))]
    stats("tamanho do modelo", sizes, "B", "E6_modelo", "E6")
    if sizes:
        print(f"  total no bucket 'models': {sum(sizes)} B  ({len(sizes)} arquivos)")
    try:
        c = supabase.table("samples").select("id", count="exact").limit(1).execute()
        print(f"  linhas em samples agora: {c.count}")
    except Exception as e:
        print(f"  contagem de samples falhou: {e}")


# --------------------------------------------------------- E3 / E4 (rede)
def _req_resp_bytes(r):
    """Tamanho aproximado na rede: linha+cabecalhos+corpo da requisicao; idem resposta."""
    req = r.request
    head = len(f"{req.method} {req.path_url} HTTP/1.1\r\n".encode())
    head += sum(len(k) + len(v) + 4 for k, v in req.headers.items()) + 2
    body = req.body or b""
    body = len(body.encode()) if isinstance(body, str) else len(body)
    rh = sum(len(k) + len(v) + 4 for k, v in r.headers.items()) + 17
    return head + body, rh + len(r.content)  # content ja decodificado; ver nota


def _rodar(nome, fn, n, rid, enlace):
    up, down, ms = [], [], []
    for _ in range(n):
        t = time.perf_counter()
        r = fn()
        ms.append((time.perf_counter() - t) * 1000)
        u, d = _req_resp_bytes(r)
        up.append(u)
        down.append(d)
    print(f"\n  {nome}  (HTTP {r.status_code})")
    stats("requisicao", up, "B", rid + "_req", enlace)
    stats("resposta", down, "B", rid + "_resp", enlace)
    stats("tempo", ms, "ms", rid + "_ms", enlace)


def medir_rest(api, n):
    import requests

    email, senha = credenciais()
    if not (email and senha):
        print("\n== E3  pulado: defina Email e Senha no .env da raiz (ou LV_EMAIL/LV_PASSWORD)")
        return
    print(f"\n== E3  REST  {api}  (tamanho conta cabecalhos; resposta decodificada)")
    _rodar("POST /auth/login", lambda: requests.post(
        f"{api}/auth/login", json={"email": email, "password": senha}, timeout=90), n, "E3_login", "E3")
    tok = requests.post(f"{api}/auth/login", json={"email": email, "password": senha},
                        timeout=90).json().get("token")
    h = {"Authorization": f"Bearer {tok}"}
    _rodar("GET /auth/profile", lambda: requests.get(f"{api}/auth/profile", headers=h, timeout=90), n, "E3_profile", "E3")
    _rodar("GET /learning/gestures", lambda: requests.get(f"{api}/learning/gestures", headers=h, timeout=90), n, "E3_gestures", "E3")


def medir_auth(n):
    import requests
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    url, anon = os.getenv("SUPABASE_URL", "").strip(), os.getenv("SUPABASE_ANON_KEY", "").strip()
    email, senha = credenciais()
    if not (url and anon and email and senha):
        print("\n== E4  pulado: precisa de SUPABASE_URL/ANON_KEY no .env e LV_EMAIL/LV_PASSWORD")
        return
    print("\n== E4  Supabase Auth (mesmas chamadas que a API faz: sign_in, refresh, get_user)")
    h = {"apikey": anon, "Content-Type": "application/json"}
    login = lambda: requests.post(f"{url}/auth/v1/token?grant_type=password",
                                  headers=h, json={"email": email, "password": senha}, timeout=60)
    _rodar("sign_in_with_password", login, n, "E4_signin", "E4")
    s = login().json()
    _rodar("get_user", lambda: requests.get(
        f"{url}/auth/v1/user", headers={**h, "Authorization": f"Bearer {s['access_token']}"}, timeout=60), n, "E4_getuser", "E4")
    # refresh invalida o token anterior: roda 1x com o refresh_token recem-obtido
    r = requests.post(f"{url}/auth/v1/token?grant_type=refresh_token", headers=h,
                      json={"refresh_token": s["refresh_token"]}, timeout=60)
    u, d = _req_resp_bytes(r)
    print(f"\n  refresh_session (1 medicao: cada refresh_token so vale 1 vez)  req={u} B  resp={d} B")


# ----------------------------------------------------------------------- log
def medir_log():
    """Frequencia real de mensagens a partir do log do app (inference-log.jsonl)."""
    import csv

    path = ROOT.parent / "Logs - APP - Diagnostico" / "inference-log.jsonl"
    print(f"\n== Log do app  {path.name}")
    ev = [json.loads(l) for l in path.open(encoding="utf8")]
    abre = [e["t"] for e in ev if e["e"] == "open"]
    fecha = [e["t"] for e in ev if e["e"] == "close"]
    res = [e for e in ev if e["e"] == "res"]
    out = ROOT / "scripts" / "log_res.csv"
    with out.open("w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["t_ms", "sessao", "frameRtt_ms", "dt_ms", "sent", "dropped"])
        for e in res:
            sess = next((i + 1 for i, (a, b) in enumerate(zip(abre, fecha)) if a <= e["t"] <= b), "")
            w.writerow([e["t"], sess, e.get("frameRtt", ""), e.get("dt", ""), e.get("sent", ""), e.get("dropped", "")])
    print("  sessoes (open_ms, close_ms, respostas):")
    for i, (a, b) in enumerate(zip(abre, fecha), 1):
        n = sum(1 for e in res if a <= e["t"] <= b)
        print(f"   {i}: {a} {b} {n}")
    total_s = sum(b - a for a, b in zip(abre, fecha)) / 1000
    print(f"  respostas={len(res)}  tempo conectado={total_s:.3f} s  taxa={len(res) / total_s:.3f} msg/s")
    cs = [e for e in ev if e["e"] == "collect_start"]
    ce = [e for e in ev if e["e"] == "collect_end" and e.get("outcome") == "saved"]
    if cs and ce:
        span = (ce[-1]["t"] - cs[0]["t"]) / 60000
        print(f"  coletas estaticas salvas={len(ce)} em {span:.2f} min -> {len(ce) / span:.2f} amostras/min "
              f"(inicio_ms={cs[0]['t']} fim_ms={ce[-1]['t']})")
        ms = [e["ms"] for e in ce]
        stats("clique -> confirmacao (WS+insert+count)", ms, "ms")
    print(f"  csv gerado: {out}")


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("alvo", choices=["ws", "e5", "e6", "rest", "auth", "log", "all"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--api", default=DEFAULT_API)
    a = ap.parse_args()
    print(f"Medicao em {time.strftime('%Y-%m-%d %H:%M:%S')}  n={a.n}")
    if a.alvo in ("ws", "all"):
        asyncio.run(medir_ws(a.api, a.n))
    if a.alvo in ("e5", "all"):
        medir_e5(a.n)
    if a.alvo in ("e6", "all"):
        medir_e6()
    if a.alvo in ("rest", "all"):
        medir_rest(a.api, a.n)
    if a.alvo in ("auth", "all"):
        medir_auth(a.n)
    if a.alvo in ("log", "all"):
        medir_log()
    if RAW:
        import csv

        out = ROOT / "scripts" / "medicoes_raw.csv"
        with out.open("w", newline="", encoding="utf8") as f:
            w = csv.writer(f)
            w.writerow(["id", "enlace", "medida", "unidade"] + [f"v{i}" for i in range(1, 11)])
            for rid, enl, lab, un, vals in RAW:
                w.writerow([rid, enl, lab, un] + [round(v) for v in vals])
        print(f"\nvalores brutos: {out}")


if __name__ == "__main__":
    main()
