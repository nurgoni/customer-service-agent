import json
import queue
import threading

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.agent.core import CustomerServiceAgent
from src.agent.events import listen
from src.models.schemas import AgentState, ChatMessage, ChatRequest, ChatResponse

router = APIRouter()


def get_agent(request: Request) -> CustomerServiceAgent:
    return request.app.state.agent


def _initial_state(payload: ChatRequest) -> AgentState:
    return AgentState(
        session_id=payload.session_id,
        user_id=payload.user_id,
        user_message=payload.message,
        conversation_history=[ChatMessage(role=m.role, content=m.content) for m in payload.history],
    )


def _to_response(state: AgentState, payload: ChatRequest) -> ChatResponse:
    return ChatResponse(
        reply=state.final_reply,
        intent=state.intent,
        products_shown=state.shown_product_ids,
        sources=state.sources,
        session_id=payload.session_id,
        trace_url=state.trace_url,
    )


@router.get("/health")
def health(request: Request):
    ctx = getattr(request.app.state, "ctx", None)
    return {
        "status": "ok",
        "vector_search": bool(ctx and ctx.vector_enabled),
        "tracing": bool(ctx and ctx.tracing_enabled),
    }


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    agent = get_agent(request)
    try:
        state = agent.run(_initial_state(payload))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return _to_response(state, payload)


@router.post("/chat/stream")
def chat_stream(payload: ChatRequest, request: Request) -> StreamingResponse:
    """
    Sama seperti /chat, tetapi mengirim langkah kerja agent secara langsung (Server-Sent Events):

        event: step     data: {"type": "intent.done", "data": {...}, "ts": ...}   (berulang)
        event: result   data: <ChatResponse>                                         (sekali, di akhir)
        event: error    data: {"detail": "..."}                                      (bila gagal)
    """
    agent = get_agent(request)
    events: queue.Queue = queue.Queue()
    done = object()

    def work() -> None:
        with listen(lambda ev: events.put(("step", ev.to_dict()))):
            try:
                state = agent.run(_initial_state(payload))
                events.put(("result", _to_response(state, payload).model_dump(mode="json")))
            except Exception as e:
                events.put(("error", {"detail": str(e)}))
            finally:
                events.put(done)

    threading.Thread(target=work, name="chat-stream", daemon=True).start()

    def sse():
        while True:
            item = events.get()
            if item is done:
                break
            kind, data = item
            yield f"event: {kind}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )