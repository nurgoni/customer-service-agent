"""
Event langkah kerja agent (bukan reasoning internal model).

Setiap langkah utama dalam satu giliran chat memancarkan event `<langkah>.start` lalu `<langkah>.done`
(atau `<langkah>.error`), misalnya:

    intent.start  -> intent.done   {intent, confidence, query, product_id, fallback, duration}
    llm.start     -> llm.done      {purpose, round, next, tools, output_tokens, duration}
    tool.start    -> tool.done     {name, args, found, count, top, error, duration}
    search.start  -> search.done   {query, method, count, top, duration}
    turn.start    -> turn.done     {message, intent, sources, products, duration}

Event hanya dikirim bila ada listener (CLI atau endpoint streaming); tanpa listener biayanya nol.
Listener disimpan di contextvar, jadi request paralel di server tidak saling tercampur.
"""

from __future__ import annotations

import contextvars
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

logger = logging.getLogger("cs_agent.events")


@dataclass
class AgentEvent:
    type: str  # mis. "intent.done"
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "data": self.data, "ts": self.ts}


Listener = Callable[[AgentEvent], None]

_listener: contextvars.ContextVar[Listener | None] = contextvars.ContextVar("agent_event_listener", default=None)


def has_listener() -> bool:
    return _listener.get() is not None


def emit(event_type: str, **data: Any) -> None:
    listener = _listener.get()
    if listener is None:
        return
    try:
        listener(AgentEvent(type=event_type, data=data))
    except Exception as e:  # tampilan tidak boleh menggagalkan agent
        logger.debug("listener event gagal: %s", e)


@contextmanager
def listen(listener: Listener | None) -> Iterator[None]:
    """Pasang listener untuk semua event di dalam blok ini (thread/konteks saat ini)."""
    token = _listener.set(listener)
    try:
        yield
    finally:
        _listener.reset(token)


@contextmanager
def step(step_name: str, /, **start_data: Any) -> Iterator[dict[str, Any]]:
    """
    Bungkus satu langkah: memancarkan `<step_name>.start`, lalu `<step_name>.done` berisi durasi dan isi dict
    yang di-yield (diisi oleh pemanggil), atau `<step_name>.error` bila terjadi exception.
    (`step_name` positional-only supaya data langkah boleh punya field bernama `name`, mis. nama tool.)
    """
    info: dict[str, Any] = {}
    emit(f"{step_name}.start", **start_data)
    started = time.perf_counter()
    try:
        yield info
    except Exception as e:
        emit(f"{step_name}.error", **start_data, error=str(e), duration=time.perf_counter() - started)
        raise
    else:
        emit(f"{step_name}.done", **start_data, **info, duration=time.perf_counter() - started)