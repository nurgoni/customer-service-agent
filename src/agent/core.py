# src/agent/core.py

import json
import sqlite3
from openai import OpenAI

from src.config import OPENAI_MODEL
from src.models.schemas import (
    AgentState, Intent, ChatMessage, ProductSearchResult
)
from src.agent.intent import classify_intent
from src.retrieval.hybrid_search import hybrid_search
from src.tools.product_tools import PRODUCT_TOOL_SCHEMAS, execute_tool


# ═══════════════════════════════════════════════════════════════
# SYSTEM PROMPTS
# ═══════════════════════════════════════════════════════════════

_SYSTEM_GENERAL = """Kamu adalah customer service AI yang ramah untuk toko online kami.
Jawab sapaan, pertanyaan umum (cara order, pengiriman, retur, dll) secara natural.
Jika customer menanyakan produk, arahkan mereka untuk bertanya lebih spesifik.
Jawab dalam Bahasa Indonesia yang sopan dan profesional."""

_SYSTEM_PRODUCT_SEARCH = """Kamu adalah customer service AI untuk toko online.
Berikut adalah produk-produk yang relevan dari katalog kami:

{product_context}

ATURAN:
1. Rekomendasikan produk HANYA dari daftar di atas.
2. Tampilkan harga PERSIS seperti data — jangan bulatkan atau ubah.
3. Jika tidak ada produk yang cocok, katakan dengan sopan.
4. Jawab dalam Bahasa Indonesia."""

_SYSTEM_EXACT_FACT = """Kamu adalah customer service AI untuk toko online.

ATURAN KRITIS:
1. SELALU gunakan tool untuk mengambil data produk.
2. JANGAN PERNAH mengarang harga, stok, atau detail produk.
3. Tampilkan harga PERSIS dari hasil tool — jangan modifikasi.
4. Jika tool mengembalikan "not found", katakan produk tidak tersedia.
5. Jawab dalam Bahasa Indonesia yang ramah."""


class CustomerServiceAgent:
    MAX_TOOL_ROUNDS = 5

    def __init__(self, client: OpenAI, conn: sqlite3.Connection):
        self.client = client
        self.conn = conn

    # ───────────────────────────────────────────────────────
    # PUBLIC
    # ───────────────────────────────────────────────────────

    def run(self, state: AgentState) -> AgentState:
        # Step 1 — Intent classification
        intent_result = classify_intent(
            self.client,
            state.user_message,
            state.conversation_history,
        )

        print(f"intent result: {intent_result}", flush=True)

        state.intent = intent_result.intent

        # Step 2 — Route
        if state.intent == Intent.GENERAL:
            state = self._lane_general(state)
        elif state.intent == Intent.PRODUCT_SEARCH:
            state = self._lane_product_search(state, intent_result.extracted_query)
        else:
            state = self._lane_exact_fact(state, intent_result.product_id)

        return state

    # ───────────────────────────────────────────────────────
    # LANE 1: GENERAL
    # ───────────────────────────────────────────────────────

    def _lane_general(self, state: AgentState) -> AgentState:
        messages = self._build_messages(state)
        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            max_tokens=512,
            messages=[{"role": "system", "content": _SYSTEM_GENERAL}] + messages,
        )
        state.final_replay = response.choices[0].message.content.strip()
        state.sources = ["llm_only"]
        return state

    # ───────────────────────────────────────────────────────
    # LANE 2: PRODUCT SEARCH (hybrid retrieval → LLM summary)
    # ───────────────────────────────────────────────────────

    def _lane_product_search(
        self, state: AgentState, query: str
    ) -> AgentState:
        results: list[ProductSearchResult] = hybrid_search(
            self.conn, query, top_k=5
        )
        state.retrieved_products = results

        if results:
            product_context = "\n\n".join(
                f"- [{r.product.id}] {r.product.name}\n"
                f"  Tipe: {r.product.product_type}\n"
                f"  Harga: Rp {r.product.price:,.0f}\n"
                f"  Stok: {r.product.stock}\n"
                f"  Deskripsi: {r.product.description}"
                for r in results
            )
            system = _SYSTEM_PRODUCT_SEARCH.format(product_context=product_context)
        else:
            system = (
                _SYSTEM_PRODUCT_SEARCH.format(product_context="(tidak ada produk yang cocok)")
            )

        messages = self._build_messages(state)
        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            max_tokens=1024,
            messages=[{"role": "system", "content": system}] + messages,
        )
        state.final_replay = response.choices[0].message.content.strip()
        state.sources = ["hybrid_search"]
        return state

    # ───────────────────────────────────────────────────────
    # LANE 3: EXACT FACT (tool calling loop)
    # ───────────────────────────────────────────────────────

    def _lane_exact_fact(
        self, state: AgentState, product_id: str | None
    ) -> AgentState:
        messages = [
            {"role": "system", "content": _SYSTEM_EXACT_FACT},
            *self._build_messages(state),
        ]

        # Jika intent classifier sudah extract product_id,
        # inject hint agar LLM tahu ID-nya
        if product_id:
            messages.append({
                "role": "system",
                "content": f"[Hint: product_id yang dimaksud user kemungkinan '{product_id}']"
            })

        for _ in range(self.MAX_TOOL_ROUNDS):
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                max_tokens=1024,
                messages=messages,
                tools=PRODUCT_TOOL_SCHEMAS,
                tool_choice="auto",
            )

            choice = response.choices[0].message

            # Jika tidak ada tool call → final answer
            if not choice.tool_calls:
                state.final_replay = (choice.content or "").strip()
                state.sources.append("db_lookup")
                return state

            # Ada tool call(s) → execute dan kirim hasilnya
            messages.append(choice)  # append assistant message with tool_calls

            for tc in choice.tool_calls:
                tool_name = tc.function.name
                tool_args = json.loads(tc.function.arguments)
                result_json = execute_tool(self.conn, tool_name, tool_args)

                state.tool_results.append({
                    "tool": tool_name,
                    "input": tool_args,
                    "output": json.loads(result_json),
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_json,
                })

        # Safety: jika loop habis tanpa final answer
        state.final_replay = "Maaf, saya mengalami kesulitan mengambil data. Silakan coba lagi."
        state.sources.append("error")
        return state

    # ───────────────────────────────────────────────────────
    # HELPERS
    # ───────────────────────────────────────────────────────

    def _build_messages(self, state: AgentState) -> list[dict]:
        msgs = [
            {"role": m.role, "content": m.content}
            for m in state.conversation_history
        ]
        msgs.append({"role": "user", "content": state.user_message})
        return msgs