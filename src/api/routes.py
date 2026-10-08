from fastapi import APIRouter, HTTPException, Request

from src.agent.core import CustomerServiceAgent
from src.models.schemas import AgentState, ChatMessage, ChatRequest, ChatResponse

router = APIRouter()


def get_agent(request: Request) -> CustomerServiceAgent:
    return request.app.state.agent


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    agent = get_agent(request)

    state = AgentState(
        session_id=payload.session_id,
        user_message=payload.message,
        conversation_history=[ChatMessage(role=m.role, content=m.content) for m in payload.history],
    )

    try:
        state = agent.run(state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ChatResponse(
        reply=state.final_reply,
        intent=state.intent,
        products_shown=state.shown_product_ids,
        sources=state.sources,
        session_id=payload.session_id,
    )
