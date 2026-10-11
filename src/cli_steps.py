"""
Tampilan langkah kerja agent di CLI: dari pesan masuk sampai jawaban siap.

Yang ditampilkan adalah langkah pipeline agent (klasifikasi intent, pencarian produk, tool call, panggilan LLM)
yang dipancarkan oleh src/agent/events.py, bukan teks reasoning internal model.

Mode:
    ringkas  satu baris per langkah (default)
    detail   ditambah kata kunci, kandidat produk + skor, argumen tool, jumlah token output
    off      tidak menampilkan langkah

Di terminal interaktif, langkah yang sedang berjalan ditampilkan dengan spinner dan timer.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

MODES = ("ringkas", "detail", "off")

INTENT_LABELS = {
    "general": "pertanyaan umum",
    "product_search": "mencari rekomendasi produk",
    "exact_fact": "info produk spesifik",
}

TOOL_LABELS = {
    "search_product_catalog": "cari katalog",
    "get_product_price": "cek harga & stok",
    "get_product_detail": "ambil detail produk",
}


def supports_unicode() -> bool:
    """Console Windows lama (conhost) sering tidak punya glyph spinner; Windows Terminal & VS Code aman."""
    return os.name != "nt" or any(k in os.environ for k in ("WT_SESSION", "TERM_PROGRAM"))


def _short(text: Any, limit: int = 40) -> str:
    text = str(text or "").replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return ""
    return f"{seconds * 1000:.0f} ms" if seconds < 0.1 else f"{seconds:.1f} dtk"


@dataclass
class _Frame:
    name: str
    data: dict[str, Any]
    started: float = field(default_factory=time.perf_counter)
    children: list[str] = field(default_factory=list)


class StepRenderer:
    def __init__(self, style, mode: str = "ringkas", unicode_ok: bool = True, out=None):
        self.style = style
        self.mode = mode if mode in MODES else "ringkas"
        self.out = out or sys.stdout
        self.live = bool(style.enabled)  # spinner butuh ANSI (hapus baris)
        if unicode_ok:
            self.sym_ok, self.sym_warn, self.sym_err, self.arrow = "✓", "⚠", "✗", "→"
            self.spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        else:
            self.sym_ok, self.sym_warn, self.sym_err, self.arrow = "+", "!", "x", "->"
            self.spinner = "|/-\\"
        self._stack: list[_Frame] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ticker: threading.Thread | None = None
        self._tick = 0

    # ------------------------------------------------------------ siklus giliran

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def begin(self) -> None:
        self._stack.clear()
        if self.enabled and self.live:
            self._stop.clear()
            self._ticker = threading.Thread(target=self._spin, daemon=True)
            self._ticker.start()

    def end(self) -> None:
        if self._ticker:
            self._stop.set()
            self._ticker.join(timeout=1)
            self._ticker = None
        with self._lock:
            # langkah yang belum selesai (mis. dibatalkan Ctrl+C) tetap ditampilkan
            while self._stack:
                frame = self._stack.pop()
                lines = [f"{self._indent()}{self.style.yellow(self.sym_warn)} {self._start_label(frame)} (tidak selesai)"]
                self._emit_lines(lines + frame.children)
            self._clear_line()

    # ------------------------------------------------------------ event

    def handle(self, event: dict[str, Any]) -> None:
        if not self.enabled:
            return
        kind = event.get("type", "")
        name, _, phase = kind.partition(".")
        if name == "turn":  # ringkasan giliran sudah ditampilkan CLI di bawah jawaban
            return
        data = event.get("data") or {}
        with self._lock:
            if phase == "start":
                self._stack.append(_Frame(name=name, data=data))
                return
            frame = self._pop(name) or _Frame(name=name, data=data)
            lines = self._format(frame, data, error=data.get("error") if phase == "error" else None)
            if lines is None:  # langkah disembunyikan pada mode ini
                lines = []
            lines = lines + frame.children
            if self._stack:
                self._stack[-1].children.extend("   " + line for line in lines)
            else:
                self._emit_lines(lines)

    # ------------------------------------------------------------ format teks

    def _start_label(self, frame: _Frame) -> str:
        d = frame.data
        if frame.name == "intent":
            return "Memahami pertanyaan"
        if frame.name == "search":
            return f'Mencari produk "{_short(d.get("query"))}"'
        if frame.name == "tool":
            return f"Mengambil data: {TOOL_LABELS.get(d.get('name'), d.get('name'))}"
        if frame.name == "llm":
            return "Menentukan langkah berikutnya" if d.get("purpose") == "tool_round" else "Menyusun jawaban"
        return frame.name

    def _format(self, frame: _Frame, d: dict[str, Any], error: str | None) -> list[str] | None:
        s, detail = self.style, self.mode == "detail"
        dur = s.dim("  " + _duration(d.get("duration")))
        ok, warn, err = s.green(self.sym_ok), s.yellow(self.sym_warn), s.red(self.sym_err)

        if error:
            return [f"{err} {self._start_label(frame)} gagal: {_short(error, 80)}{dur}"]

        if frame.name == "intent":
            label = INTENT_LABELS.get(d.get("intent"), d.get("intent"))
            if d.get("fallback"):
                lines = [f"{warn} Memahami pertanyaan {self.arrow} {d['fallback']}, dianggap {label}{dur}"]
            else:
                conf = d.get("confidence")
                conf_txt = f" (yakin {conf:.0%})" if isinstance(conf, (int, float)) else ""
                lines = [f"{ok} Memahami pertanyaan {self.arrow} {label}{conf_txt}{dur}"]
            if detail and not d.get("fallback"):
                extra = f'kata kunci: "{_short(d.get("query"), 60)}"'
                if d.get("product_id"):
                    extra += f" · product_id: {d['product_id']}"
                lines.append(s.dim(f"    {extra}"))
            return lines

        if frame.name == "search":
            nested = bool(self._stack)  # pencarian di dalam tool: sudah terwakili oleh baris tool
            if nested and not detail:
                return None
            method = "hybrid" if str(d.get("method", "")).startswith("hybrid") else "kata kunci"
            lines = [f'{ok} Mencari produk "{_short(d.get("query"))}" {self.arrow} {d.get("count", 0)} kandidat ({method}){dur}']
            if detail:
                for i, p in enumerate(d.get("top") or [], 1):
                    lines.append(s.dim(f"    {i}. {p['name']} · {p['price']} · skor {p['score']}"))
            return lines

        if frame.name == "tool":
            label = TOOL_LABELS.get(d.get("name"), d.get("name"))
            args = d.get("args") or {}
            arg_txt = ", ".join(f'"{_short(v, 30)}"' for v in args.values())
            top = d.get("top") or []
            if not d.get("found"):
                summary = "tidak ditemukan"
                sym = warn
            elif d.get("count", 0) > 1:
                summary = f"{d['count']} produk, teratas {top[0]['name']} ({top[0]['price']})" if top else f"{d['count']} produk"
                sym = ok
            else:
                p = top[0] if top else {}
                summary = f"{p.get('name')} · {p.get('price')} · stok {p.get('stock')}"
                sym = ok
            lines = [f"{sym} Mengambil data: {label}({arg_txt}) {self.arrow} {summary}{dur}"]
            # daftar kandidat cukup ditampilkan sekali: bila pencarian di dalam tool sudah menampilkannya, lewati
            if detail and d.get("count", 0) > 1 and not frame.children:
                for i, p in enumerate(top, 1):
                    lines.append(s.dim(f"    {i}. {p['name']} · {p['price']} · stok {p['stock']}"))
            return lines

        if frame.name == "llm":
            tokens = d.get("output_tokens")
            token_txt = f" · {tokens} token" if detail and tokens is not None else ""
            if d.get("purpose") == "tool_round" and d.get("next") == "tools":
                tools = ", ".join(TOOL_LABELS.get(t, t) for t in d.get("tools") or [])
                return [f"{ok} Menentukan langkah {self.arrow} perlu data: {tools}{token_txt}{dur}"]
            if d.get("purpose") == "tool_round":
                return [f"{ok} Menyusun jawaban dari data produk{token_txt}{dur}"]
            return [f"{ok} Menyusun jawaban{token_txt}{dur}"]

        return [f"{ok} {frame.name}{dur}"]

    # ------------------------------------------------------------ output

    def _pop(self, name: str) -> _Frame | None:
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i].name == name:
                return self._stack.pop(i)
        return None

    def _indent(self) -> str:
        return "   " * len(self._stack)

    def _clear_line(self) -> None:
        if self.live:
            self.out.write("\r\033[K")
            self.out.flush()

    def _emit_lines(self, lines: list[str]) -> None:
        self._clear_line()
        for line in lines:
            self.out.write(line + "\n")
        self.out.flush()

    def _spin(self) -> None:
        while not self._stop.wait(0.1):
            with self._lock:
                if not self._stack:
                    continue
                frame = self._stack[-1]
                self._tick += 1
                glyph = self.spinner[self._tick % len(self.spinner)]
                elapsed = time.perf_counter() - frame.started
                indent = "   " * (len(self._stack) - 1)
                text = f"{indent}{self.style.cyan(glyph)} {self._start_label(frame)}… {self.style.dim(_duration(elapsed))}"
                self.out.write("\r\033[K" + text)
                self.out.flush()