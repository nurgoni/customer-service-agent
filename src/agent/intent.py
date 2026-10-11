import json
import logging
import re

from openai import OpenAI

from src.agent.events import step
from src.config import LLM_EXTRA, OPENAI_MODEL_FAST
from src.models.schemas import ChatMessage, Intent, IntentResult
from src.observability import llm_kwargs, observe, update_span

logger = logging.getLogger("cs_agent.intent")

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


@observe(name="intent-classification", as_type="chain")
def classify_intent(client: OpenAI, user_message: str, history: list[ChatMessage]) -> IntentResult:
    with step("intent") as info:
        result, fallback_reason = _classify(client, user_message, history)
        info.update(
            intent=result.intent.value,
            confidence=result.confidence,
            query=result.extracted_query,
            product_id=result.product_id,
            fallback=fallback_reason,
        )
        return result


def _classify(client: OpenAI, user_message: str, history: list[ChatMessage]) -> tuple[IntentResult, str | None]:
    """Return (hasil intent, alasan fallback atau None bila klasifikasi berhasil)."""
    update_span(input={"message": user_message, "history_used": min(len(history), 3)})

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
        **llm_kwargs("intent-classifier"),
    )

    choice = response.choices[0]
    raw = clean_llm_text(choice.message.content)
    logger.info("model=%s finish_reason=%s raw=%r", response.model, choice.finish_reason, raw)

    if not raw:
        logger.warning("content kosong -> fallback general (cek thinking mode / num_ctx model)")
        result = _fallback(user_message)
        update_span(output=result.model_dump(mode="json"), level="WARNING", status_message="empty classifier output")
        return result, "output classifier kosong"

    try:
        data = _extract_json(raw)
        product_id = data.get("product_id")
        if not product_id or str(product_id).lower() in ("null", "none"):
            product_id = None
        confidence = min(max(float(data.get("confidence", 0.8)), 0.0), 1.0)
        result = IntentResult(
            intent=Intent(str(data["intent"]).strip().lower()),
            confidence=confidence,
            extracted_query=data.get("extracted_query") or user_message,
            product_id=product_id,
        )
        update_span(output=result.model_dump(mode="json"))
        return result, None
    except (json.JSONDecodeError, KeyError, ValueError, TypeError, AttributeError) as e:
        logger.warning("gagal parse (%s) -> fallback general", e)
        result = _fallback(user_message)
        update_span(output=result.model_dump(mode="json"), level="WARNING", status_message=f"parse error: {e}")
        return result, "output classifier tidak bisa dibaca"