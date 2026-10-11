"""
Chat dengan customer service agent lewat command line.

Pemakaian (dari folder root project):
    python -m src.cli                               # agent dijalankan langsung, tanpa server
    python -m src.cli --api http://localhost:8000   # lewat server API yang sedang berjalan
    python -m src.cli --steps detail                # tampilkan langkah kerja agent secara lengkap
    python -m src.cli --verbose                     # tampilkan log teknis (intent, tool call, dll)
    python -m src.cli --user budi --session demo-1  # set user_id / session_id (terlihat di Langfuse)

Perintah di dalam chat:
    /help     daftar perintah
    /steps    ganti tampilan langkah: ringkas -> detail -> off (atau /steps <mode>)
    /reset    hapus riwayat dan mulai session baru
    /history  tampilkan riwayat percakapan
    /meta     tampilkan/sembunyikan info intent, sumber, dan link trace
    /exit     keluar (atau Ctrl+C / Ctrl+D)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from contextlib import nullcontext
from typing import Any, Callable

from src.cli_steps import MODES as STEP_MODES
from src.cli_steps import StepRenderer, supports_unicode

MAX_HISTORY = 20  # sama dengan batas field `history` di API

EventCallback = Callable[[dict[str, Any]], None]


# ------------------------------------------------------------------ tampilan

class Style:
    enabled = False

    @classmethod
    def wrap(cls, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if cls.enabled else text

    @classmethod
    def bold(cls, t: str) -> str:
        return cls.wrap("1", t)

    @classmethod
    def dim(cls, t: str) -> str:
        return cls.wrap("2", t)

    @classmethod
    def green(cls, t: str) -> str:
        return cls.wrap("32", t)

    @classmethod
    def cyan(cls, t: str) -> str:
        return cls.wrap("36", t)

    @classmethod
    def red(cls, t: str) -> str:
        return cls.wrap("31", t)

    @classmethod
    def yellow(cls, t: str) -> str:
        return cls.wrap("33", t)


def _setup_console(no_color: bool) -> None:
    # Hindari UnicodeEncodeError di console Windows yang memakai code page lama.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    if os.name == "nt":
        os.system("")  # mengaktifkan dukungan warna ANSI di console Windows
    Style.enabled = sys.stdout.isatty() and not no_color and "NO_COLOR" not in os.environ


HELP_TEXT = """Perintah:
  /help     daftar perintah
  /steps    ganti tampilan langkah: ringkas -> detail -> off (atau /steps ringkas|detail|off)
  /reset    hapus riwayat dan mulai session baru
  /history  tampilkan riwayat percakapan
  /meta     tampilkan/sembunyikan info intent, sumber, dan link trace
  /exit     keluar (atau Ctrl+C / Ctrl+D)"""


# ------------------------------------------------------------------ backend

class LocalBackend:
    """Menjalankan agent langsung di proses ini (tidak perlu server API)."""

    def __init__(self, verbose: bool):
        from src.bootstrap import build_app_context, setup_logging

        setup_logging("INFO" if verbose else "WARNING")
        self.ctx = build_app_context()

    def describe(self) -> str:
        search = "hybrid (BM25 + vector)" if self.ctx.vector_enabled else "BM25 saja"
        if not self.ctx.tracing_enabled:
            tracing = "nonaktif"
        elif self.ctx.tracing_connected:
            tracing = "aktif"
        else:
            tracing = "aktif, tetapi server tidak terhubung"
        return f"mode lokal · pencarian: {search} · Langfuse: {tracing}"

    def send(
        self,
        message: str,
        history: list[dict],
        session_id: str,
        user_id: str | None,
        on_event: EventCallback | None = None,
    ) -> dict[str, Any]:
        from src import observability
        from src.agent.events import listen
        from src.models.schemas import AgentState, ChatMessage

        state = AgentState(
            session_id=session_id,
            user_id=user_id,
            user_message=message,
            conversation_history=[ChatMessage(**m) for m in history],
        )
        listener = listen(lambda ev: on_event(ev.to_dict())) if on_event else nullcontext()
        with listener:
            state = self.ctx.agent.run(state)
        if self.ctx.tracing_connected:
            # Trace langsung muncul di Langfuse tanpa menunggu batch. Dilewati bila server tidak terhubung
            # agar chat tidak tertahan oleh percobaan ulang pengiriman.
            observability.flush()
        return {
            "reply": state.final_reply,
            "intent": state.intent.value if state.intent else None,
            "sources": state.sources,
            "products_shown": state.shown_product_ids,
            "trace_url": state.trace_url,
        }

    def close(self) -> None:
        from src.bootstrap import close_app_context

        close_app_context(self.ctx)


class ApiBackend:
    """Mengirim pesan ke server API (`uvicorn src.main:app`) yang sedang berjalan."""

    def __init__(self, base_url: str, timeout: float):
        import httpx

        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(base_url=self.base_url, timeout=timeout)
        self.streaming = True  # dimatikan otomatis bila server belum punya /chat/stream
        try:
            health = self.client.get("/api/v1/health")
            health.raise_for_status()
            self.health = health.json()
        except Exception as e:
            self.client.close()
            raise RuntimeError(
                f"Server API di {self.base_url} tidak bisa dihubungi ({e}). "
                "Jalankan dulu: uvicorn src.main:app"
            ) from e

    def describe(self) -> str:
        search = "hybrid (BM25 + vector)" if self.health.get("vector_search") else "BM25 saja"
        tracing = "aktif" if self.health.get("tracing") else "nonaktif"
        return f"mode API ({self.base_url}) · pencarian: {search} · Langfuse: {tracing}"

    def send(
        self,
        message: str,
        history: list[dict],
        session_id: str,
        user_id: str | None,
        on_event: EventCallback | None = None,
    ) -> dict[str, Any]:
        payload = {"message": message, "history": history, "session_id": session_id, "user_id": user_id}
        if on_event and self.streaming:
            result = self._send_stream(payload, on_event)
            if result is not None:
                return result
        return self._send_plain(payload)

    def _send_plain(self, payload: dict) -> dict[str, Any]:
        response = self.client.post("/api/v1/chat", json=payload)
        if response.status_code >= 400:
            raise RuntimeError(f"HTTP {response.status_code}: {self._detail(response)}")
        return response.json()

    def _send_stream(self, payload: dict, on_event: EventCallback) -> dict[str, Any] | None:
        """Server-Sent Events dari /api/v1/chat/stream. Return None bila server tidak mendukung streaming."""
        result: dict[str, Any] | None = None
        with self.client.stream("POST", "/api/v1/chat/stream", json=payload) as response:
            if response.status_code in (404, 405):
                self.streaming = False
                return None
            if response.status_code >= 400:
                response.read()
                raise RuntimeError(f"HTTP {response.status_code}: {self._detail(response)}")

            event_name, data_lines = "message", []
            for line in response.iter_lines():
                if line.startswith("event:"):
                    event_name = line[6:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].strip())
                elif line == "" and data_lines:
                    data = json.loads("\n".join(data_lines))
                    if event_name == "step":
                        on_event(data)
                    elif event_name == "result":
                        result = data
                    elif event_name == "error":
                        raise RuntimeError(data.get("detail", "error tidak diketahui"))
                    event_name, data_lines = "message", []
        if result is None:
            raise RuntimeError("koneksi streaming terputus sebelum jawaban diterima")
        return result

    @staticmethod
    def _detail(response) -> str:
        try:
            return str(response.json().get("detail"))
        except Exception:
            return response.text

    def close(self) -> None:
        self.client.close()


# ------------------------------------------------------------------ loop chat

def _new_session_id() -> str:
    return f"cli-{uuid.uuid4().hex[:8]}"


def _print_meta(data: dict[str, Any], elapsed: float) -> None:
    parts = [f"intent: {data.get('intent')}", f"sumber: {', '.join(data.get('sources') or []) or '-'}"]
    if data.get("products_shown"):
        parts.append(f"produk: {', '.join(data['products_shown'])}")
    parts.append(f"{elapsed:.1f} dtk")
    print(Style.dim("  (" + " · ".join(parts) + ")"))
    if data.get("trace_url"):
        print(Style.dim(f"  trace: {data['trace_url']}"))


def _handle_command(message: str, state: dict[str, Any], renderer: StepRenderer, history: list[dict]) -> bool:
    """Proses perintah /xxx. Return False bila pengguna ingin keluar."""
    parts = message.split()
    command = parts[0].lower()
    if command in ("/exit", "/quit", "/keluar"):
        return False
    if command == "/help":
        print(HELP_TEXT)
    elif command == "/steps":
        if len(parts) > 1 and parts[1].lower() in STEP_MODES:
            renderer.mode = parts[1].lower()
        elif len(parts) > 1:
            print(Style.yellow(f"  mode tidak dikenal: {parts[1]} (pilih: {', '.join(STEP_MODES)})"))
            return True
        else:
            renderer.mode = STEP_MODES[(STEP_MODES.index(renderer.mode) + 1) % len(STEP_MODES)]
        print(Style.dim(f"  tampilan langkah: {renderer.mode}"))
    elif command == "/reset":
        history.clear()
        state["session_id"] = _new_session_id()
        print(Style.dim(f"  riwayat dihapus · session baru: {state['session_id']}"))
    elif command == "/history":
        if not history:
            print(Style.dim("  (riwayat masih kosong)"))
        for m in history:
            label = "Anda " if m["role"] == "user" else "Agent"
            print(f"  {label}: {m['content']}")
    elif command == "/meta":
        state["show_meta"] = not state["show_meta"]
        print(Style.dim(f"  info meta {'ditampilkan' if state['show_meta'] else 'disembunyikan'}"))
    else:
        print(Style.yellow(f"  perintah tidak dikenal: {command} (ketik /help)"))
    return True


def run_chat(backend, session_id: str, user_id: str | None, show_meta: bool, renderer: StepRenderer) -> None:
    history: list[dict] = []
    state = {"session_id": session_id, "show_meta": show_meta}
    while True:
        try:
            message = input(Style.bold(Style.green("Anda  > "))).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not message:
            continue
        if message.startswith("/"):
            if not _handle_command(message, state, renderer, history):
                break
            continue

        if not renderer.enabled and Style.enabled:
            print(Style.dim("Agent > sedang memproses..."), end="\r", flush=True)
        started = time.perf_counter()
        renderer.begin()
        try:
            data = backend.send(
                message,
                history[-MAX_HISTORY:],
                state["session_id"],
                user_id,
                on_event=renderer.handle if renderer.enabled else None,
            )
        except KeyboardInterrupt:
            renderer.end()
            print(Style.yellow("  dibatalkan"))
            continue
        except Exception as e:
            renderer.end()
            if Style.enabled:
                print("\r\033[K", end="")
            print(Style.red(f"  Gagal memproses pesan: {e}"))
            continue
        renderer.end()
        elapsed = time.perf_counter() - started

        if not renderer.enabled and Style.enabled:
            print("\r\033[K", end="")
        reply = data.get("reply", "")
        print(f"{Style.bold(Style.cyan('Agent > '))}{reply}")
        if state["show_meta"]:
            _print_meta(data, elapsed)
        print()

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": reply})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.cli", description="Chat dengan customer service agent.")
    parser.add_argument("--api", metavar="URL", help="pakai server API yang sedang berjalan, mis. http://localhost:8000")
    parser.add_argument("--session", metavar="ID", help="session_id (default: dibuat acak, mis. cli-1a2b3c4d)")
    parser.add_argument("--user", metavar="ID", help="user_id pelanggan, dicatat di Langfuse")
    parser.add_argument(
        "--steps",
        choices=STEP_MODES,
        default="ringkas",
        help="tampilan langkah kerja agent: ringkas (default), detail, atau off",
    )
    parser.add_argument("--timeout", type=float, default=300.0, help="batas waktu request mode API, detik (default 300)")
    parser.add_argument("--verbose", "-v", action="store_true", help="tampilkan log teknis agent (mode lokal)")
    parser.add_argument("--no-meta", action="store_true", help="sembunyikan info intent/sumber di bawah jawaban")
    parser.add_argument("--no-color", action="store_true", help="matikan warna dan animasi")
    args = parser.parse_args(argv)

    _setup_console(args.no_color)
    session_id = args.session or _new_session_id()
    renderer = StepRenderer(Style, mode=args.steps, unicode_ok=supports_unicode())

    print(Style.dim("Menyiapkan agent..."))
    try:
        backend = ApiBackend(args.api, args.timeout) if args.api else LocalBackend(args.verbose)
    except Exception as e:
        print(Style.red(f"Gagal menyiapkan agent: {e}"))
        return 1

    print(Style.bold("Customer Service Agent - chat CLI"))
    print(Style.dim(backend.describe()))
    print(Style.dim(f"session: {session_id} · langkah: {renderer.mode} · ketik /help untuk daftar perintah, /exit untuk keluar\n"))

    try:
        run_chat(backend, session_id, args.user, show_meta=not args.no_meta, renderer=renderer)
    finally:
        backend.close()
    print(Style.dim("Sampai jumpa!"))
    return 0


if __name__ == "__main__":
    sys.exit(main())