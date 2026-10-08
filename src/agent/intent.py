import json
import re

from openai import OpenAI

from src.config import LLM_EXTRA, OPENAI_MODEL_FAST
from src.models.schemas import ChatMessage, Intent, IntentResult

_CLASSIFIER_PROMPT = """Kamu adalah pengklasifikasi intent untuk agent layanan pelanggan toko online.
Klasifikasikan pesan pengguna ke salah satu dari tiga kategori:

- "general": sapaan, pertanyaan umum (cara order, pengiriman, retur), atau di luar topik produk.
- "product_search": pengguna mencari/meminta rekomendasi produk berdasarkan kategori, rentang harga, atau kebutuhan.
- "exact_fact": pengguna menanyakan harga/stok/detail SATU produk spesifik yang nama atau ID-nya disebut.

Balas HANYA dengan satu objek JSON dengan field:
{"intent": "general|product_search|exact_fact", "confidence": 0.0-1.0, "extracted_query": "kata kunci pencarian singkat", "product_id": "ID seperti SKU001 jika disebut, atau null"}
"""

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def clean_llm_text(text: str | None) -> str:
    """Buang blok <think>...</think> yang kadang ikut di content pada model thinking (Qwen3 dkk)."""
    text = _THINK_BLOCK.sub("", text or "")
    # blok <think> yang terpotong (tanpa penutup) -> tidak ada jawaban yang bisa dipakai
    if "<think>" in text.lower():
        text = text[: text.lower().index("<think>")]
    return text.strip()


def _extract_json(raw: str) -> dict:
    """Ambil objek JSON pertama dari teks (toleran terhadap code fence / kalimat pembuka)."""
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _fallback(user_message: str) -> IntentResult:
    return IntentResult(intent=Intent.GENERAL, confidence=0.0, extracted_query=user_message, product_id=None)


def classify_intent(client: OpenAI, user_message: str, history: list[ChatMessage]) -> IntentResult:
    history_snippet = ""
    if history:
        recent = "\n".join(f"{m.role.upper()}: {m.content}" for m in history[-3:])
        history_snippet = f"Recent conversation:\n{recent}\n\n"

    # Catatan: classifier TIDAK diberi `tools` - tugasnya hanya mengembalikan JSON.
    response = client.chat.completions.create(
        model=OPENAI_MODEL_FAST,
        max_tokens=512,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _CLASSIFIER_PROMPT},
            {"role": "user", "content": f"{history_snippet}User message: {user_message}"},
        ],
        **LLM_EXTRA,
    )

    choice = response.choices[0]
    raw = clean_llm_text(choice.message.content)
    print(f"[intent] model={response.model} finish_reason={choice.finish_reason} raw={raw!r}", flush=True)

    if not raw:
        print("[intent] content kosong -> fallback general (cek thinking mode / num_ctx model)", flush=True)
        return _fallback(user_message)

    try:
        data = _extract_json(raw)
        product_id = data.get("product_id")
        if not product_id or str(product_id).lower() in ("null", "none"):
            product_id = None
        confidence = min(max(float(data.get("confidence", 0.8)), 0.0), 1.0)
        return IntentResult(
            intent=Intent(str(data["intent"]).strip().lower()),
            confidence=confidence,
            extracted_query=data.get("extracted_query") or user_message,
            product_id=product_id,
        )
    except (json.JSONDecodeError, KeyError, ValueError, TypeError, AttributeError) as e:
        print(f"[intent] gagal parse ({e}) -> fallback general", flush=True)
        return _fallback(user_message)
