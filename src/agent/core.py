import json
import sqlite3
import anthropic

from src.models.schemas import (
    AgentState, Intent, ChatMessage, ProductSearchResult
)
from src.agent.intent import classify_intent
from src.retrieval.hybrid_search import hybrid_search
from src.tools.product_tools import PRODUCT_TOOL_SCHEMAS, execute_tool


class CustomerServiceAgent:
    """
    
    """

    MAX_TOOL_ITERATIONS = 5

    def __init__(
        self,
        client: anthropic.Anthropic,
        conn: sqlite3.Connection
    ):
        self.client = client
        self.conn = conn

    def run(self, state: AgentState) -> AgentState:
        """
        
        """

        # --- step 1: Intent Classification ---
        intent_result = classify_intent(
            self.client,
            self.user_message,
            state.conversation_history
        )
        state.intent = intent_result.intent

        # --- step 2: Route to correct lane ---
        if state.intent == Intent.GENERAL:
            state = self._lane_general(state)
        elif state.intent == Intent.PRODUCT_SEARCH:
            state = self._lane_product_search(
                state,
                intent_result.extracted_query
            )
        else:
            product_id = intent_result.product_id
            state = self._lane_exact_fact(state, product_id)
        
        return state

    def _lane_general(self, state: AgentState) -> AgentState:
        messages = self._build_messsages(state)
        response = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            system=_SYSTEM_GENERAL,
            messages=messages,
        )
        state.final_replay = response.content[0].text.strip()
        state.sources = ["llm_only"]
        return state

    def _lane_product_search(
        self,
        state: AgentState,
        query: str
    ) -> AgentState:
        """
        
        """
        # retrieve top 5 relevant products via hybrid search
        results: list[ProductSearchResult] = hybrid_search(
            self.conn,
            query,
            top_k=5
        )
        state.retrieved_products = results

        # Build a product context block to inject into the prompt
        if results:
            product_context = "\n\n".join(

            )
            system_with_context = (

            )
        else:
            system_with_context = (

            )

        
