"""
Diagnosis throughput LLM di Ollama untuk customer service agent.

Mengukur di mesin Anda sendiri:
  1. Apakah model mendukung thinking, dan apakah LLM_REASONING_EFFORT benar-benar mematikannya
  2. Kecepatan: waktu ke token pertama (TTFT) dan token/detik saat generate
  3. Apakah model termuat penuh di GPU atau sebagian jatuh ke CPU
  4. Apakah model jawaban, model intent, dan model embedding bisa dimuat bersamaan (tanpa bongkar-pasang)

Pemakaian (dari folder root project, Ollama harus berjalan):
    python scripts/diagnose_llm.py
    python scripts/diagnose_llm.py --runs 3      # ulangi pengukuran agar lebih stabil
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402
from openai import OpenAI  # noqa: E402

from src.config import (  # noqa: E402
    EMBEDDING_MODEL,
    LLM_EXTRA,
    LLM_REASONING_EFFORT,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    OPENAI_MODEL_FAST,
)

PROMPT = [
    {"role": "system", "content": "Kamu customer service toko online. Jawab dalam Bahasa Indonesia, maksimal 3 kalimat."},
    {"role": "user", "content": "Saya mau beli sepatu lari untuk pemula, apa yang perlu saya perhatikan?"},
]


def native_root() -> str:
    base = (OPENAI_BASE_URL or "").rstrip("/")
    return base[:-3] if base.endswith("/v1") else base


def header(title: str) -> None:
    print(f"\n=== {title}")


def show_model(http: httpx.Client, model: str) -> None:
    try:
        info = http.post("/api/show", json={"model": model}).json()
    except Exception as e:
        print(f"  {model}: gagal membaca /api/show ({e})")
        return
    caps = info.get("capabilities") or []
    print(f"  {model}: capabilities={caps}")
    if "thinking" in info:
        print(f"    metadata thinking: {info['thinking']}")
    params = info.get("parameters") or ""
    ctx = [line for line in params.splitlines() if "num_ctx" in line]
    if ctx:
        print(f"    {ctx[0].strip()}  (diatur di Modelfile, menimpa OLLAMA_CONTEXT_LENGTH)")


def measure(client: OpenAI, model: str, extra: dict, runs: int) -> dict:
    """Streaming agar TTFT dan kecepatan generate bisa dipisahkan."""
    results = []
    for _ in range(runs):
        started = time.perf_counter()
        first = None
        content, reasoning, usage = "", "", None
        stream = client.chat.completions.create(
            model=model,
            messages=PROMPT,
            max_tokens=1024,
            temperature=0,
            stream=True,
            stream_options={"include_usage": True},
            **extra,
        )
        for chunk in stream:
            if chunk.usage:
                usage = chunk.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            extra_fields = getattr(delta, "model_extra", None) or {}
            piece_reasoning = extra_fields.get("reasoning") or extra_fields.get("reasoning_content") or ""
            piece = delta.content or ""
            if (piece or piece_reasoning) and first is None:
                first = time.perf_counter()
            content += piece
            reasoning += piece_reasoning
        total = time.perf_counter() - started
        completion = usage.completion_tokens if usage else None
        gen_time = total - (first - started) if first else None
        results.append(
            {
                "total": total,
                "ttft": (first - started) if first else None,
                "completion_tokens": completion,
                "tok_per_s": (completion / gen_time) if completion and gen_time else None,
                "reasoning_chars": len(reasoning) + (len(content) - len(content.split("</think>")[-1])),
                "answer_chars": len(content.split("</think>")[-1].strip()),
            }
        )
    # run pertama bisa termasuk waktu memuat model; laporkan run terakhir sebagai kondisi "hangat"
    return {"first": results[0], "warm": results[-1]}


def fmt(r: dict) -> str:
    def f(v, unit="", digits=1):
        return f"{v:.{digits}f}{unit}" if isinstance(v, (int, float)) else "-"

    return (
        f"total {f(r['total'], ' dtk')}, TTFT {f(r['ttft'], ' dtk', 2)}, "
        f"token output {r['completion_tokens'] if r['completion_tokens'] is not None else '-'}, "
        f"{f(r['tok_per_s'], ' tok/dtk')}, teks thinking {r['reasoning_chars']} karakter, jawaban {r['answer_chars']} karakter"
    )


def check_loaded(http: httpx.Client) -> None:
    try:
        models = http.get("/api/ps").json().get("models", [])
    except Exception as e:
        print(f"  gagal membaca /api/ps ({e})")
        return
    if not models:
        print("  tidak ada model yang termuat")
        return
    for m in models:
        size, vram = m.get("size") or 0, m.get("size_vram") or 0
        pct = 100 * vram / size if size else 0
        flag = "OK" if pct >= 99 else "PERINGATAN: sebagian di CPU -> generate jauh lebih lambat"
        print(f"  {m.get('name')}: {size / 1e9:.1f} GB, {pct:.0f}% di GPU, kedaluwarsa {m.get('expires_at')}  [{flag}]")
    names = {m.get("name") for m in models}
    needed = dict.fromkeys([OPENAI_MODEL, OPENAI_MODEL_FAST, EMBEDDING_MODEL])
    missing = [n for n in needed if n not in names and f"{n}:latest" not in names]
    if missing:
        print(f"  PERINGATAN: belum/tidak lagi termuat bersamaan: {missing}. Bila VRAM tidak cukup, Ollama")
        print("  membongkar-pasang model di setiap giliran chat (intent -> embedding -> jawaban).")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=2, help="jumlah pengukuran per skenario (default 2)")
    args = parser.parse_args()

    if not OPENAI_BASE_URL:
        print("OPENAI_BASE_URL kosong: script ini untuk Ollama.")
        return 1

    client = OpenAI(api_key=OPENAI_API_KEY or "ollama", base_url=OPENAI_BASE_URL)
    http = httpx.Client(base_url=native_root(), timeout=600)

    header("1. Kemampuan model (/api/show)")
    for model in dict.fromkeys([OPENAI_MODEL, OPENAI_MODEL_FAST, EMBEDDING_MODEL]):
        show_model(http, model)

    header(f"2. Model jawaban '{OPENAI_MODEL}' dengan pengaturan aplikasi (reasoning_effort={LLM_REASONING_EFFORT or '-'})")
    with_app = measure(client, OPENAI_MODEL, LLM_EXTRA, max(args.runs, 1))
    print("  run pertama :", fmt(with_app["first"]))
    print("  run terakhir:", fmt(with_app["warm"]))

    header("3. Pembanding: tanpa parameter reasoning_effort (perilaku default model)")
    default = measure(client, OPENAI_MODEL, {}, 1)["warm"]
    print("  ", fmt(default))

    w = with_app["warm"]
    if w["reasoning_chars"] > 0:
        print("\n  >> THINKING MASIH AKTIF dengan pengaturan aplikasi. Setiap panggilan LLM menghasilkan token thinking")
        print("     yang tidak dipakai. Ini kemungkinan bottleneck terbesar.")
    elif w["completion_tokens"] and default["completion_tokens"] and default["completion_tokens"] > 2 * w["completion_tokens"]:
        print("\n  >> reasoning_effort bekerja: tanpa parameter itu model menghasilkan jauh lebih banyak token.")
    else:
        print("\n  >> Tidak terdeteksi thinking pada pengaturan aplikasi.")
    if w["completion_tokens"] and w["answer_chars"] and w["completion_tokens"] > 3 * (w["answer_chars"] / 3.5):
        print("  >> Catatan: token output jauh lebih banyak dari panjang jawaban; kemungkinan ada thinking tersembunyi.")

    if OPENAI_MODEL_FAST != OPENAI_MODEL:
        header(f"4. Model intent '{OPENAI_MODEL_FAST}'")
        print("  ", fmt(measure(client, OPENAI_MODEL_FAST, LLM_EXTRA, max(args.runs, 1))["warm"]))

    header(f"5. Embedding '{EMBEDDING_MODEL}'")
    try:
        started = time.perf_counter()
        client.embeddings.create(model=EMBEDDING_MODEL, input=["sepatu lari ringan"])
        print(f"  {time.perf_counter() - started:.2f} dtk per query embedding")
    except Exception as e:
        print(f"  gagal: {e}")

    header("6. Model yang termuat sekarang (/api/ps)")
    check_loaded(http)
    return 0


if __name__ == "__main__":
    sys.exit(main())