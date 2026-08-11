# src/agent/intent.py

import json
import re
from openai import OpenAI

from src.config import OPENAI_API_KEY, OPENAI_MODEL_FAST
from src.models.schemas import Intent, IntentResult, ChatMessage


_CLASSIFIER_PROMPT = """Kamu adalah AI yang bertugas untuk mendeteksi kategori pesan dari user.
Setiap pesan masuk dari user pasti akan terdiri dari tiga kategori dibawah ini:

- "general": pertanyaan umum, sapaan, atau diluar topik mengenai produk.
- "product_search": pengguna mencari produk berdasarkan kategori, kebutuhan, atau deskripsi umum.
- "exact_fact": pengguna menanyakan detail atau harga satu produk berdasarkan nama produk tersebut.

response output yang kamu berikan harus dalam bentuk JSON dengan fields:

intent: kategori chat dari user.
confidence: tingkat keyakinan anda terhadap prediksi kategori.
extracted_query: raw chat dari user.
product_id: Nama dari produk yang sedang dicari detail atau harga-nya. Jika chat bukan kategori "exact_fact", maka nilainya adalah None.

contoh response jika kategori chat user adalah general:
{
    "intent": "general", 
    "confidence": 0.95, 
    "extracted_query": "Bagaimana kabarmu hari ini", 
    "product_id": None
}
"""


def classify_intent(
    client: OpenAI,
    user_message: str,
    history: list[ChatMessage],
) -> IntentResult:
    history_snippet = ""
    if history:
        recent = history[-3:]
        history_snippet = "\n".join(
            f"{m.role.upper()}: {m.content}" for m in recent
        )
        history_snippet = f"\nRecent conversation:\n{history_snippet}\n"

    response = client.chat.completions.create(
        model=OPENAI_MODEL_FAST,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": _CLASSIFIER_PROMPT},
            {"role": "user", "content": f"{history_snippet}User message: {user_message}"},
        ],
        extra_body={"think": False}
    )

    raw = response.choices[0].message.content.strip()
    # raw = response.choices[0].message

    # ── DEBUG: print SEBELUM parsing, pakai flush=True ──
    # print(f"[DEBUG] raw content dari OpenAI: {repr(raw)}", flush=True)

    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    print(f"[DEBUG] raw content dari OpenAI: {repr(raw)}", flush=True)

    try:
        data = json.loads(raw)

        print(f"[DEBUG] parsed intent: {data}", flush=True)

        return IntentResult(
            intent=Intent(data["intent"]),
            confidence=float(data.get("confidence", 0.8)),
            extracted_query=data.get("extracted_query", user_message),
            product_id=data.get("product_id"),
        )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        # ── DEBUG: cetak error aslinya, jangan silent ──
        print(f"[DEBUG] ⚠️ Parsing gagal! Error: {e}", flush=True)
        print(f"[DEBUG] Raw yang gagal di-parse: {repr(raw)}", flush=True)

        return IntentResult(
            intent=Intent.GENERAL,
            confidence=0.0,
            extracted_query=user_message,
            product_id=None,
        )