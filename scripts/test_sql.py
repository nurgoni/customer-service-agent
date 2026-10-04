"""
=============================================================
 SQL Capability Testing — Qwen via Ollama (OpenAI-compatible)
=============================================================

Apa yang dilakukan script ini:
  1. Terhubung ke Ollama lewat OpenAI SDK (base_url lokal)
  2. Menjalankan serangkaian soal SQL dengan tingkat kesulitan
     berbeda (Basic → Advanced)
  3. Setiap soal berjalan dalam *satu sesi multi-turn* yang benar:
       system  → menetapkan konteks & persona evaluator
       user    → menyajikan soal
       assistant → dijawab model
       user    → follow-up atau soal berikutnya
       …dst.
  4. Mencetak hasil lengkap ke terminal beserta skor akhir.

Persyaratan:
  pip install openai
  ollama pull qwen2.5-coder:7b   (atau model lain yang kamu pilih)
  ollama serve                    (pastikan berjalan di port 11434)
"""

from __future__ import annotations

import textwrap
import time
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

# ─────────────────────────────────────────────
# 1. KONFIGURASI
# ─────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY  = "ollama"          # nilai apa pun diterima Ollama
MODEL_NAME      = "qwen3.5:9b"   # ganti sesuai model yang sudah di-pull

client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key=OLLAMA_API_KEY,
)

# ─────────────────────────────────────────────
# 2. DEFINISI SOAL SQL
# ─────────────────────────────────────────────

@dataclass
class SQLQuestion:
    """Satu soal pengujian SQL beserta kunci evaluasinya."""
    id:         int
    level:      str          # "Basic" | "Intermediate" | "Advanced"
    topic:      str
    question:   str
    followup:   str          # pertanyaan lanjutan setelah jawaban pertama
    keywords:   list[str]    # kata kunci yang *harus* ada di jawaban
    bonus_keys: list[str] = field(default_factory=list)  # bonus poin


SQL_QUESTIONS: list[SQLQuestion] = [
    # ── BASIC ────────────────────────────────
    SQLQuestion(
        id=1, level="Basic", topic="SELECT & WHERE",
        question=textwrap.dedent("""\
            Diberikan tabel berikut:

            CREATE TABLE employees (
                emp_id   INT PRIMARY KEY,
                name     VARCHAR(100),
                dept     VARCHAR(50),
                salary   DECIMAL(10,2),
                hire_date DATE
            );

            Tuliskan query SQL untuk menampilkan nama dan gaji
            semua karyawan di departemen 'Engineering' yang
            gajinya di atas 5.000.000, diurutkan dari gaji
            tertinggi ke terendah."""),
        followup="Bagaimana jika kita ingin menambahkan kolom dept juga, "
                 "dan hanya menampilkan 5 baris teratas?",
        keywords=["SELECT", "WHERE", "ORDER BY"],
        bonus_keys=["LIMIT", "TOP"],
    ),

    SQLQuestion(
        id=2, level="Basic", topic="Aggregate Functions",
        question=textwrap.dedent("""\
            Masih menggunakan tabel employees di atas.
            Tuliskan query untuk menghitung:
            - Jumlah karyawan per departemen
            - Rata-rata gaji per departemen
            - Hanya tampilkan departemen dengan rata-rata gaji > 4.000.000"""),
        followup="Apa perbedaan mendasar antara WHERE dan HAVING dalam SQL?",
        keywords=["GROUP BY", "COUNT", "AVG", "HAVING"],
    ),

    # ── INTERMEDIATE ─────────────────────────
    SQLQuestion(
        id=3, level="Intermediate", topic="JOIN",
        question=textwrap.dedent("""\
            Diberikan dua tabel:

            CREATE TABLE departments (
                dept_id   INT PRIMARY KEY,
                dept_name VARCHAR(50),
                manager_id INT           -- referensi ke emp_id
            );

            -- employees.dept sekarang menyimpan dept_id (INT)

            Tuliskan query yang menampilkan:
            nama karyawan, nama departemen, dan nama manajer
            departemen mereka.
            Gunakan JOIN yang sesuai dan jelaskan alasan pemilihan
            jenis JOIN tersebut."""),
        followup="Apa yang terjadi jika seorang karyawan belum memiliki "
                 "departemen? JOIN mana yang lebih tepat digunakan?",
        keywords=["JOIN", "ON", "LEFT", "INNER"],
        bonus_keys=["alias", "self join", "LEFT JOIN"],
    ),

    SQLQuestion(
        id=4, level="Intermediate", topic="Subquery & CTE",
        question=textwrap.dedent("""\
            Tuliskan query untuk menemukan karyawan yang gajinya
            di atas rata-rata gaji departemennya sendiri.
            Berikan DUA solusi:
            1. Menggunakan correlated subquery
            2. Menggunakan CTE (Common Table Expression)"""),
        followup="Dari sisi performa, mana yang biasanya lebih efisien "
                 "dan mengapa? Apakah jawabannya selalu sama di semua RDBMS?",
        keywords=["WITH", "AS", "SELECT", "AVG", "subquery"],
        bonus_keys=["correlated", "CTE", "performance", "execution plan"],
    ),

    # ── ADVANCED ─────────────────────────────
    SQLQuestion(
        id=5, level="Advanced", topic="Window Functions",
        question=textwrap.dedent("""\
            Tuliskan query yang menghasilkan laporan berikut
            untuk setiap karyawan:
            - nama, departemen, gaji
            - ranking gaji dalam departemennya (1 = tertinggi)
            - persentase gaji terhadap total gaji departemen
            - gaji kumulatif dalam departemen (diurutkan hire_date)

            Gunakan window functions dan jelaskan setiap fungsi
            yang kamu pakai."""),
        followup="Apa perbedaan antara RANK(), DENSE_RANK(), dan ROW_NUMBER()? "
                 "Berikan contoh kasus di mana hasilnya berbeda.",
        keywords=["OVER", "PARTITION BY", "RANK", "SUM", "ORDER BY"],
        bonus_keys=["DENSE_RANK", "ROW_NUMBER", "ROWS BETWEEN", "FRAME"],
    ),

    SQLQuestion(
        id=6, level="Advanced", topic="Query Optimization",
        question=textwrap.dedent("""\
            Query berikut berjalan sangat lambat pada tabel
            employees dengan 10 juta baris:

            SELECT e.name, e.salary, d.dept_name
            FROM employees e
            JOIN departments d ON e.dept = d.dept_id
            WHERE YEAR(e.hire_date) = 2023
              AND e.salary > (SELECT AVG(salary) FROM employees)
            ORDER BY e.salary DESC;

            Identifikasi minimal 3 masalah performa dan berikan
            solusi konkret untuk masing-masing (termasuk DDL
            index jika diperlukan)."""),
        followup="Bagaimana cara membaca dan menginterpretasikan EXPLAIN / "
                 "EXPLAIN ANALYZE untuk query di atas?",
        keywords=["INDEX", "CREATE INDEX", "function", "sargable"],
        bonus_keys=["covering index", "materialized", "execution plan",
                    "statistics", "partition"],
    ),
]

# ─────────────────────────────────────────────
# 3. SISTEM PROMPT (PERAN EVALUATOR)
# ─────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
    Kamu adalah seorang database engineer senior dengan pengalaman
    lebih dari 10 tahun di bidang SQL dan desain sistem basis data.
    Kamu sedang mengikuti sesi evaluasi kemampuan SQL.

    Aturan penting:
    - Berikan jawaban SQL yang valid dan siap dijalankan.
    - Sertakan penjelasan singkat tapi jelas untuk setiap keputusan.
    - Jika ada beberapa pendekatan, sebutkan trade-off-nya.
    - Gunakan bahasa Indonesia untuk penjelasan, SQL tetap dalam bahasa standar.
    - Jangan menambahkan informasi yang tidak diminta.
""")

# ─────────────────────────────────────────────
# 4. FUNGSI INTI: SATU SESI MULTI-TURN
# ─────────────────────────────────────────────

def run_sql_test_session(q: SQLQuestion) -> dict:
    """
    Menjalankan satu soal SQL dalam satu sesi multi-turn yang benar.

    Struktur conversation history yang dikelola:
    ┌──────────────┬────────────────────────────────────────────────┐
    │ Turn         │ Role       │ Isi                               │
    ├──────────────┼────────────┼───────────────────────────────────┤
    │ (init)       │ system     │ SYSTEM_PROMPT (konteks evaluasi)  │
    │ Turn 1 →     │ user       │ Soal SQL                          │
    │ Turn 1 ←     │ assistant  │ Jawaban model                     │
    │ Turn 2 →     │ user       │ Follow-up question                │
    │ Turn 2 ←     │ assistant  │ Jawaban follow-up                 │
    └──────────────┴────────────┴───────────────────────────────────┘

    Model SELALU menerima seluruh conversation_history sehingga
    jawaban follow-up memiliki konteks penuh (stateless API,
    stateful history di sisi klien).
    """
    print(f"\n{'═'*65}")
    print(f"  [{q.id}/{len(SQL_QUESTIONS)}] {q.level} — {q.topic}")
    print(f"{'═'*65}")

    # ── conversation_history adalah satu-satunya "memori" sesi ──
    conversation_history: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    # ── TURN 1: soal utama ───────────────────────────────────────
    conversation_history.append({"role": "user", "content": q.question})
    print(f"\n[USER → MODEL]\n{q.question}\n")

    t0 = time.perf_counter()
    response1 = client.chat.completions.create(
        model=MODEL_NAME,
        messages=conversation_history,  # kirim SELURUH history
        temperature=0.2,                # rendah agar SQL deterministik
        max_tokens=1024,
    )
    elapsed1 = time.perf_counter() - t0

    answer1 = response1.choices[0].message.content
    # Tambahkan jawaban model ke history — WAJIB untuk multi-turn
    conversation_history.append({"role": "assistant", "content": answer1})

    print(f"[MODEL → USER]  ({elapsed1:.1f}s)\n{answer1}\n")

    # ── TURN 2: follow-up ────────────────────────────────────────
    conversation_history.append({"role": "user", "content": q.followup})
    print(f"[USER → MODEL] (follow-up)\n{q.followup}\n")

    t0 = time.perf_counter()
    response2 = client.chat.completions.create(
        model=MODEL_NAME,
        messages=conversation_history,  # context lengkap: system + T1 + T2
        temperature=0.2,
        max_tokens=512,
    )
    elapsed2 = time.perf_counter() - t0

    answer2 = response2.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": answer2})

    print(f"[MODEL → USER] (follow-up answer)  ({elapsed2:.1f}s)\n{answer2}\n")

    # ── EVALUASI SEDERHANA (keyword matching) ────────────────────
    full_text = (answer1 + " " + answer2).upper()
    hits   = sum(1 for kw in q.keywords   if kw.upper() in full_text)
    bonuses = sum(1 for kw in q.bonus_keys if kw.upper() in full_text)

    score = (hits / len(q.keywords)) * 100 if q.keywords else 0
    print(f"  ✓ Keyword hits : {hits}/{len(q.keywords)} "
          f"({score:.0f}%)  |  Bonus: {bonuses}/{len(q.bonus_keys)}")

    return {
        "id":            q.id,
        "level":         q.level,
        "topic":         q.topic,
        "score_pct":     score,
        "bonus_hits":    bonuses,
        "time_total_s":  round(elapsed1 + elapsed2, 1),
        "turns":         len([m for m in conversation_history
                              if m["role"] != "system"]),
    }


# ─────────────────────────────────────────────
# 5. RINGKASAN AKHIR
# ─────────────────────────────────────────────

def print_summary(results: list[dict]) -> None:
    print(f"\n{'═'*65}")
    print("  RINGKASAN HASIL PENGUJIAN SQL")
    print(f"{'═'*65}")
    print(f"  Model : {MODEL_NAME}")
    print(f"  Soal  : {len(results)} pertanyaan (multi-turn)")
    print(f"{'─'*65}")
    print(f"  {'#':<4} {'Level':<14} {'Topik':<26} {'Score':>6} {'Bonus':>6} {'Waktu':>7}")
    print(f"  {'─'*4} {'─'*14} {'─'*26} {'─'*6} {'─'*6} {'─'*7}")

    total_score = 0
    for r in results:
        total_score += r["score_pct"]
        print(f"  {r['id']:<4} {r['level']:<14} {r['topic']:<26} "
              f"{r['score_pct']:>5.0f}% {r['bonus_hits']:>5}  "
              f"{r['time_total_s']:>6.1f}s")

    avg = total_score / len(results) if results else 0
    print(f"{'─'*65}")
    print(f"  Rata-rata keyword score  : {avg:.1f}%")
    print(f"  Total waktu              : "
          f"{sum(r['time_total_s'] for r in results):.1f}s")
    print(f"{'═'*65}\n")

    # Interpretasi
    if avg >= 80:
        verdict = "🟢 SANGAT BAIK — model memahami SQL secara komprehensif."
    elif avg >= 60:
        verdict = "🟡 CUKUP — model menguasai SQL dasar-menengah, " \
                  "tapi lemah di konsep lanjutan."
    else:
        verdict = "🔴 PERLU PENINGKATAN — model kesulitan dengan SQL."
    print(f"  Verdict: {verdict}\n")


# ─────────────────────────────────────────────
# 6. ENTRY POINT
# ─────────────────────────────────────────────

def main() -> None:
    print(f"\n{'═'*65}")
    print("  SQL CAPABILITY TEST — Ollama × OpenAI SDK")
    print(f"  Model : {MODEL_NAME}")
    print(f"  URL   : {OLLAMA_BASE_URL}")
    print(f"{'═'*65}")

    # Cek koneksi cepat
    try:
        probe = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
        )
        print(f"\n  ✅ Koneksi berhasil. Model merespons: "
              f"'{probe.choices[0].message.content.strip()[:40]}'\n")
    except Exception as exc:
        print(f"\n  ❌ Gagal terhubung ke Ollama: {exc}")
        print("     Pastikan 'ollama serve' sudah berjalan dan model sudah di-pull.")
        return

    results: list[dict] = []
    for q in SQL_QUESTIONS:
        try:
            result = run_sql_test_session(q)
            results.append(result)
        except Exception as exc:
            print(f"  ⚠️  Soal {q.id} gagal: {exc}")
            results.append({
                "id": q.id, "level": q.level, "topic": q.topic,
                "score_pct": 0, "bonus_hits": 0, "time_total_s": 0,
                "turns": 0,
            })

    print_summary(results)


if __name__ == "__main__":
    main()