"""
Observability dengan Langfuse - sepenuhnya opsional.

- Bila LANGFUSE_PUBLIC_KEY dan LANGFUSE_SECRET_KEY diisi, setiap giliran chat menjadi satu trace di Langfuse:
  intent classifier, jalur yang dipilih, hybrid search, tool call, dan setiap panggilan LLM
  (model, prompt, jawaban, token, latensi) tercatat sebagai observation bertingkat.
- Bila tidak diisi, semua fungsi di modul ini menjadi no-op dan library `langfuse` tidak di-import sama sekali.

Kegagalan Langfuse (server mati, key salah) tidak boleh menghentikan agent: semua pemanggilan dibungkus try/except.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Callable, Iterator, TypeVar

from src.config import LANGFUSE_BASE_URL, LANGFUSE_ENABLED, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY

logger = logging.getLogger("cs_agent.observability")

F = TypeVar("F", bound=Callable[..., Any])

_langfuse = None

if LANGFUSE_ENABLED:
    from langfuse import Langfuse, is_default_export_span
    from langfuse import observe as _lf_observe
    from langfuse import propagate_attributes as _lf_propagate_attributes
    from langfuse.openai import OpenAI as _OpenAIClient  # drop-in: tiap panggilan LLM & embedding tercatat

    # Wrapper Langfuse mencatat SEMUA panggilan OpenAI SDK di proses ini, termasuk cek embedding saat startup.
    # Panggilan embedding yang tidak berada di dalam giliran chat (tanpa parent) dibuang agar tidak
    # menjadi trace "OpenAI-embedding" yang berdiri sendiri setiap kali server dinyalakan.
    _DROP_ROOT_SPANS = {"OpenAI-embedding"}

    def _should_export_span(span) -> bool:
        if span.parent is None and span.name in _DROP_ROOT_SPANS:
            return False
        return is_default_export_span(span)

    # Inisialisasi eksplisit agar SDK memakai server lokal, bukan default Langfuse Cloud.
    # Instance pertama ini juga yang dipakai oleh wrapper OpenAI dan decorator @observe.
    _langfuse = Langfuse(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        base_url=LANGFUSE_BASE_URL,
        should_export_span=_should_export_span,
    )
else:
    from openai import OpenAI as _OpenAIClient


# Batas Langfuse untuk atribut yang dipropagasi (session_id, user_id, nilai metadata): string <= 200 karakter.
_MAX_ATTR_LEN = 200


def _clip(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)[:_MAX_ATTR_LEN]


def is_enabled() -> bool:
    return LANGFUSE_ENABLED


def create_openai_client(**kwargs: Any):
    """Client OpenAI SDK. Saat tracing aktif, memakai wrapper Langfuse yang mencatat setiap panggilan LLM."""
    return _OpenAIClient(**kwargs)


def llm_kwargs(name: str) -> dict:
    """Argumen tambahan untuk chat.completions.create: nama generation di Langfuse (diabaikan bila tracing mati)."""
    return {"name": name} if LANGFUSE_ENABLED else {}


def observe(name: str | None = None, as_type: str = "span") -> Callable[[F], F]:
    """
    Decorator untuk membuat observation (span) di sekitar sebuah fungsi.

    Input/output fungsi TIDAK ditangkap otomatis karena argumennya berisi koneksi DB atau client.
    Isi input/output yang bermakna lewat `update_span(...)` dari dalam fungsi.
    """

    def decorator(fn: F) -> F:
        if not LANGFUSE_ENABLED:
            return fn
        return _lf_observe(
            name=name or fn.__name__,
            as_type=as_type,
            capture_input=False,
            capture_output=False,
        )(fn)

    return decorator


@contextmanager
def trace_attributes(
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    trace_name: str | None = None,
) -> Iterator[None]:
    """Atribut trace (session, user, tags, metadata) untuk semua observation di dalam blok ini."""
    if not LANGFUSE_ENABLED:
        yield
        return
    with _lf_propagate_attributes(
        session_id=_clip(session_id),
        user_id=_clip(user_id),
        tags=[_clip(t) for t in tags] if tags else None,
        metadata={k: _clip(v) for k, v in metadata.items()} if metadata else None,
        trace_name=trace_name,
    ):
        yield


def update_span(**kwargs: Any) -> None:
    """Perbarui observation yang sedang aktif (input, output, metadata, level, status_message, name)."""
    if not LANGFUSE_ENABLED:
        return
    try:
        _langfuse.update_current_span(**kwargs)
    except Exception as e:  # observability tidak boleh menggagalkan agent
        logger.debug("update_span gagal: %s", e)


def set_trace_io(*, input: Any = None, output: Any = None) -> None:
    """Isi input/output level trace (yang tampil di daftar trace Langfuse)."""
    if not LANGFUSE_ENABLED:
        return
    try:
        _langfuse.set_current_trace_io(input=input, output=output)
    except Exception as e:
        logger.debug("set_trace_io gagal: %s", e)


def current_trace_url() -> str | None:
    """URL trace yang sedang aktif di UI Langfuse (None bila tracing mati)."""
    if not LANGFUSE_ENABLED:
        return None
    try:
        return _langfuse.get_trace_url(trace_id=_langfuse.get_current_trace_id())
    except Exception:
        return None


def check_connection() -> bool | None:
    """True bila key valid dan server bisa dihubungi, False bila gagal, None bila tracing mati."""
    if not LANGFUSE_ENABLED:
        return None
    try:
        return bool(_langfuse.auth_check())
    except Exception as e:
        logger.warning("Langfuse tidak bisa dihubungi di %s: %s", LANGFUSE_BASE_URL, e)
        return False


def flush() -> None:
    """Kirim semua trace yang masih di buffer (dipanggil per giliran di CLI dan saat shutdown)."""
    if not LANGFUSE_ENABLED:
        return
    try:
        _langfuse.flush()
    except Exception as e:
        logger.debug("flush gagal: %s", e)


def shutdown() -> None:
    if not LANGFUSE_ENABLED:
        return
    try:
        _langfuse.shutdown()
    except Exception as e:
        logger.debug("shutdown gagal: %s", e)