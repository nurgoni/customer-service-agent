import json
import sqlite3

from openai import OpenAI

from src.agent.intent import classify_intent, clean_llm_text
from src.config import LLM_EXTRA, OPENAI_MODEL
from src.models.schemas import AgentState, Intent, ProductSearchResult
from src.retrieval.hybrid_search import hybrid_search
from src.tools.product_tools import PRODUCT_TOOL_SCHEMAS, execute_tool

_SYSTEM_GENERAL = """Kamu adalah customer service AI yang ramah untuk toko online kami.
Jawab sapaan dan pertanyaan umum (cara order, pengiriman, retur, dll) secara natural.
Kamu TIDAK punya data harga/stok produk di mode ini; jika customer menanyakan produk, minta mereka menyebut nama atau kebutuhan produknya.
Jangan mengarang kebijakan toko yang tidak kamu ketahui; sarankan menghubungi tim kami bila ragu.
Jawab dalam Bahasa Indonesia yang sopan dan ringkas."""

_SYSTEM_PRODUCT_SEARCH = """Kamu adalah customer service AI untuk toko online.
Berikut produk-produk relevan dari katalog resmi kami:

{product_context}

ATURAN:
1. Rekomendasikan produk HANYA dari daftar di atas.
2. Tulis harga PERSIS seperti pada data (format Rupiah yang sama) - jangan dibulatkan atau diubah.
3. Sebutkan jika stok 0 (habis).
4. Jika tidak ada produk yang cocok, katakan dengan sopan bahwa produk tidak ada di katalog.
5. Jawab dalam Bahasa Indonesia yang ramah dan ringkas."""

_SYSTEM_EXACT_FACT = """Kamu adalah customer service AI untuk toko online.

ATURAN KRITIS:
1. SELALU gunakan tool untuk mengambil data produk. Jika hanya tahu nama produk, panggil search_product_catalog dulu.
2. JANGAN PERNAH mengarang harga, stok, atau detail produk.
3. Tulis harga PERSIS seperti field formatted_price dari tool - jangan diubah.
4. Jika tool mengembalikan found=false, katakan produk tidak tersedia di katalog.
5. Jika ada beberapa produk mirip, tanyakan produk mana yang dimaksud atau sebutkan semuanya.
6. Jawab dalam Bahasa Indonesia yang ramah dan ringkas."""

_EMPTY_REPLY_FALLBACK = "Maaf, saya belum bisa menjawab saat ini. Silakan coba lagi sebentar lagi."


class CustomerServiceAgent:
    MAX_TOOL_ROUNDS = 5

    def __init__(self, client: OpenAI, conn: sqlite3.Connection):
        self.client = client
        self.conn = conn

    # ---------------------------------------------------------------- public

    def run(self, state: AgentState) -> AgentState:
        intent_result = classify_intent(self.client, state.user_message, state.conversation_history)
        state.intent = intent_result.intent

        if state.intent == Intent.GENERAL:
            return self._lane_general(state)
        if state.intent == Intent.PRODUCT_SEARCH:
            return self._lane_product_search(state, intent_result.extracted_query)
        return self._lane_exact_fact(state, intent_result.product_id)

    # ----------------------------------------------------------- lane general

    def _lane_general(self, state: AgentState) -> AgentState:
        text = self._chat(
            [{"role": "system", "content": _SYSTEM_GENERAL}, *self._build_messages(state)],
            max_tokens=1024,
        )
        state.final_reply = text or _EMPTY_REPLY_FALLBACK
        state.sources = ["llm_only"]
        return state

    # ---------------------------------------------------- lane product search

    def _lane_product_search(self, state: AgentState, query: str) -> AgentState:
        results: list[ProductSearchResult] = hybrid_search(self.conn, query, top_k=5)
        state.retrieved_products = results
        state.shown_product_ids = [r.product.id for r in results]

        if results:
            product_context = "\n\n".join(
                f"- [{r.product.id}] {r.product.name}\n"
                f"  Tipe: {r.product.product_type}\n"
                f"  Harga: {r.product.price_display()}\n"
                f"  Stok: {r.product.stock}\n"
                f"  Deskripsi: {r.product.description}"
                for r in results
            )
        else:
            product_context = "(tidak ada produk yang cocok di katalog)"

        text = self._chat(
            [
                {"role": "system", "content": _SYSTEM_PRODUCT_SEARCH.format(product_context=product_context)},
                *self._build_messages(state),
            ],
            max_tokens=1024,
        )
        state.final_reply = text or _EMPTY_REPLY_FALLBACK
        state.sources = ["hybrid_search"]
        return state

    # ------------------------------------------------------ lane exact fact

    def _lane_exact_fact(self, state: AgentState, product_id: str | None) -> AgentState:
        system = _SYSTEM_EXACT_FACT
        if product_id:
            # Hint digabung ke system prompt: pesan system di tengah percakapan tidak selalu didukung model lokal.
            system += f"\n\nPetunjuk: product_id yang dimaksud customer kemungkinan '{product_id}'."

        messages: list[dict] = [{"role": "system", "content": system}, *self._build_messages(state)]

        for _ in range(self.MAX_TOOL_ROUNDS):
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                max_tokens=1024,
                temperature=0,
                messages=messages,
                tools=PRODUCT_TOOL_SCHEMAS,
                **LLM_EXTRA,
            )
            message = response.choices[0].message

            if not message.tool_calls:
                state.final_reply = clean_llm_text(message.content) or _EMPTY_REPLY_FALLBACK
                if "db_lookup" not in state.sources:
                    state.sources.append("db_lookup")
                return state

            # Simpan giliran assistant sebagai dict biasa agar aman untuk endpoint kompatibel-OpenAI (Ollama).
            messages.append(
                {
                    "role": "assistant",
                    "content": clean_llm_text(message.content),
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"},
                        }
                        for tc in message.tool_calls
                    ],
                }
            )

            for tc in message.tool_calls:
                tool_name = tc.function.name
                try:
                    tool_args = json.loads(tc.function.arguments or "{}")
                    if not isinstance(tool_args, dict):
                        tool_args = {}
                except json.JSONDecodeError:
                    tool_args = {}
                result_json = execute_tool(self.conn, tool_name, tool_args)
                result = json.loads(result_json)
                print(f"[agent] tool={tool_name} args={tool_args}", flush=True)

                state.tool_results.append({"tool": tool_name, "input": tool_args, "output": result})
                self._collect_product_ids(state, result)

                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_json})

        state.final_reply = "Maaf, saya mengalami kesulitan mengambil data produk. Silakan coba lagi."
        state.sources.append("error")
        return state

    # --------------------------------------------------------------- helpers

    def _chat(self, messages: list, max_tokens: int) -> str:
        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            max_tokens=max_tokens,
            temperature=0.3,
            messages=messages,
            **LLM_EXTRA,
        )
        choice = response.choices[0]
        text = clean_llm_text(choice.message.content)
        if not text:
            print(f"[agent] jawaban kosong finish_reason={choice.finish_reason}", flush=True)
        return text

    @staticmethod
    def _collect_product_ids(state: AgentState, result: dict) -> None:
        ids: list[str] = []
        if isinstance(result.get("products"), list):
            ids += [p["id"] for p in result["products"] if "id" in p]
        if isinstance(result.get("product"), dict) and "id" in result["product"]:
            ids.append(result["product"]["id"])
        if result.get("found") and result.get("product_id"):
            ids.append(result["product_id"])
        for pid in ids:
            if pid not in state.shown_product_ids:
                state.shown_product_ids.append(pid)

    @staticmethod
    def _build_messages(state: AgentState) -> list[dict]:
        msgs = [{"role": m.role, "content": m.content} for m in state.conversation_history]
        msgs.append({"role": "user", "content": state.user_message})
        return msgs
