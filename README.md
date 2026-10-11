# Customer Service Agent

Chatbot customer service yang menjawab pertanyaan pelanggan tentang produk (nama, harga, stok, deskripsi)
berdasarkan data produk Anda sendiri. LLM dijalankan lokal lewat **Ollama**, data disimpan di **SQLite**,
agent bisa diakses lewat **REST API (FastAPI)** atau **chat di command line**, dan setiap percakapan bisa
dipantau di **Langfuse**.

Prinsip utamanya: **harga dan stok tidak pernah dikarang oleh LLM**. Angka selalu diambil dari database
dan diformat oleh kode; LLM hanya merangkai kalimat jawabannya.

## Daftar isi

1. [Cara kerja](#1-cara-kerja)
2. [Prasyarat](#2-prasyarat)
3. [Instalasi dari nol](#3-instalasi-dari-nol)
4. [Konfigurasi `.env`](#4-konfigurasi-env)
5. [Menyiapkan data produk](#5-menyiapkan-data-produk)
6. [Chat lewat command line](#6-chat-lewat-command-line)
7. [Menjalankan server API](#7-menjalankan-server-api)
8. [Memakai API](#8-memakai-api)
9. [Observability dengan Langfuse](#9-observability-dengan-langfuse)
10. [Mengganti data atau model](#10-mengganti-data-atau-model)
11. [Memakai OpenAI sebagai pengganti Ollama](#11-memakai-openai-sebagai-pengganti-ollama)
12. [Struktur project](#12-struktur-project)
13. [Troubleshooting](#13-troubleshooting)
14. [Batasan](#14-batasan)

---

## 1. Cara kerja

Setiap pesan pelanggan melewati dua tahap: klasifikasi intent, lalu salah satu dari tiga jalur.

```
Pesan pelanggan (CLI atau API)
      |
      v
Intent classifier (LLM, output JSON)
      |
      +-- general ---------> LLM menjawab langsung (sapaan, cara order, dll)
      |
      +-- product_search --> Hybrid search (BM25 + vector) -> LLM merangkum hasil
      |
      +-- exact_fact ------> LLM memanggil tool -> query SQLite -> LLM menyusun jawaban
      |
      v
Jawaban  (+ satu trace di Langfuse bila tracing aktif)
```

| Intent | Contoh pertanyaan | Sumber jawaban (`sources`) |
|---|---|---|
| `general` | "Halo, gimana cara order?" | `llm_only` |
| `product_search` | "Ada headphone yang baterainya awet?" | `hybrid_search` |
| `exact_fact` | "Berapa harga Nike Air Max?" | `db_lookup` |

Hybrid search menggabungkan pencarian kata kunci (BM25) dan pencarian makna (embedding) dengan
Reciprocal Rank Fusion. Jika pencarian vektor tidak tersedia, sistem otomatis memakai BM25 saja.

---

## 2. Prasyarat

| Kebutuhan | Keterangan |
|---|---|
| Python | Versi 3.10 atau lebih baru (`python --version`) |
| Ollama | Terpasang di komputer (https://ollama.com/download) **atau** dijalankan lewat Docker (`compose.yaml`) |
| Model LLM | Model yang mendukung *tool calling*, mis. keluarga Qwen. Default project: `qwen3.8:27b` |
| Model embedding | `bge-m3:567m` |
| Docker Desktop | Diperlukan untuk Langfuse (dan Ollama versi Docker). Opsional bila tidak memakai observability |
| Hardware | Model 27B butuh GPU/RAM besar. Jika mesin terbatas, pakai varian yang lebih kecil dan sesuaikan `.env` |

Project ini dikembangkan di Windows. Perintah untuk Linux/macOS disertakan di tiap langkah.

---

## 3. Instalasi dari nol

Semua perintah dijalankan dari **folder root project**, yaitu folder yang berisi `src/`, `compose.yaml`,
dan `requirements.txt`.

### Langkah 1 - Ambil kode

Ekstrak project (atau clone repository), lalu masuk ke foldernya:

```powershell
cd customer_service_agent
```

### Langkah 2 - Buat virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Langkah 3 - Install dependency

```powershell
pip install -r requirements.txt
```

### Langkah 4 - Buat file `.env`

Windows:

```powershell
copy .env.example .env
```

Linux/macOS:

```bash
cp .env.example .env
```

File `.env` dibaca oleh aplikasi Python **dan** oleh Docker Compose. Penjelasan tiap variabel ada di
[bagian 4](#4-konfigurasi-env). Untuk percobaan lokal, nilai bawaan sudah bisa dipakai; ganti nilai
bertanda `CHANGEME` sebelum dipakai bersama orang lain.

### Langkah 5 - Jalankan layanan Docker (Langfuse dan, bila perlu, Ollama)

Pilih salah satu sesuai cara Anda menjalankan Ollama:

```powershell
# A. Ollama di Docker + Langfuse
docker compose up -d

# B. Ollama sudah terpasang di komputer -> jalankan Langfuse saja
docker compose up -d langfuse-web langfuse-worker
```

Pada opsi B, layanan pendukung Langfuse (Postgres, ClickHouse, Redis, MinIO) ikut berjalan otomatis.
Jangan menjalankan opsi A bila Ollama di komputer sedang aktif, karena keduanya memakai port 11434.

Start pertama butuh beberapa menit untuk mengunduh image. Cek statusnya:

```powershell
docker compose ps
```

Langfuse siap ketika http://localhost:3000 menampilkan halaman login. Login dengan
`LANGFUSE_INIT_USER_EMAIL` dan `LANGFUSE_INIT_USER_PASSWORD` dari `.env`. Organisasi, project, dan API
key sudah dibuat otomatis, jadi tidak ada yang perlu dibuat manual di UI.

Tidak ingin memakai Langfuse? Lewati langkah ini (atau jalankan hanya `docker compose up -d ollama`)
dan set `LANGFUSE_ENABLED=false` di `.env`.

### Langkah 6 - Siapkan model Ollama

Ollama di komputer:

```powershell
ollama pull qwen3.8:27b
ollama pull bge-m3:567m
ollama list
```

Ollama di Docker:

```powershell
docker exec -it ollama ollama pull qwen3.8:27b
docker exec -it ollama ollama pull bge-m3:567m
docker exec -it ollama ollama list
```

Catat nama model persis seperti yang tampil di `ollama list`; nama itu yang diisi di `.env`.

Bila memakai Ollama di komputer, naikkan context window-nya (Ollama versi Docker sudah diatur lewat
`OLLAMA_CONTEXT_LENGTH` di `.env`). Nilai default-nya kecil, sehingga prompt berisi daftar produk bisa
terpotong tanpa peringatan.

```powershell
setx OLLAMA_CONTEXT_LENGTH 8192
```

Lalu tutup dan jalankan ulang aplikasi Ollama. Di Linux/macOS: `OLLAMA_CONTEXT_LENGTH=8192 ollama serve`.

Pastikan Ollama aktif dengan membuka http://localhost:11434 di browser (harus tampil `Ollama is running`).

### Langkah 7 - Muat data produk ke database

```powershell
python -m src.db.seed
```

Perintah ini membaca `data/products.csv` (data contoh), mengisi tabel `products`, dan membuat embedding.
Output yang diharapkan:

```
[seed] 8 produk dimasukkan ke tabel products
[seed] membuat embedding dengan model 'bge-m3:567m'...
[seed] 8 embedding tersimpan (dimensi 1024)
```

Untuk memakai data Anda sendiri, lihat [bagian 5](#5-menyiapkan-data-produk).

### Langkah 8 - Mulai chat

Cara tercepat untuk mencoba agent adalah lewat command line:

```powershell
python -m src.cli
```

Untuk aplikasi lain (web, WhatsApp, dll), jalankan server API: lihat [bagian 7](#7-menjalankan-server-api).

---

## 4. Konfigurasi `.env`

### Aplikasi

| Variabel | Contoh | Keterangan |
|---|---|---|
| `OPENAI_API_KEY` | `ollama` | Wajib diisi. Ollama tidak memeriksa nilainya, jadi isi bebas asal tidak kosong |
| `OPENAI_BASE_URL` | `http://localhost:11434/v1` | Alamat Ollama. Akhiran `/v1` wajib ada |
| `OPENAI_MODEL` | `qwen3.8:27b` | Model untuk menyusun jawaban dan memanggil tool |
| `OPENAI_MODEL_FAST` | `qwen3.8:27b` | Model untuk intent classifier. Boleh model lebih kecil agar lebih cepat |
| `LLM_REASONING_EFFORT` | `none` | Mematikan *thinking mode* pada model seperti Qwen3. Kosongkan untuk tidak mengirim parameter ini |
| `EMBEDDING_MODEL` | `bge-m3:567m` | Model embedding di Ollama |
| `EMBEDDING_BASE_URL` | (kosong) | Opsional. Isi hanya bila embedding dilayani server lain; default-nya sama dengan `OPENAI_BASE_URL` |
| `OLLAMA_CONTEXT_LENGTH` | `8192` | Context window untuk Ollama yang berjalan di Docker |
| `LOG_LEVEL` | `INFO` | Level log aplikasi: `DEBUG`, `INFO`, atau `WARNING` |

Nama variabel memakai awalan `OPENAI_` karena project ini berkomunikasi dengan Ollama lewat
library `openai` (Ollama menyediakan endpoint yang kompatibel).

### Langfuse: dipakai aplikasi

| Variabel | Contoh | Keterangan |
|---|---|---|
| `LANGFUSE_ENABLED` | `true` | Set `false` untuk mematikan tracing tanpa menghapus key |
| `LANGFUSE_PUBLIC_KEY` | `pk-lf-customer-service-local` | API key project. Tracing hanya aktif bila public **dan** secret key terisi |
| `LANGFUSE_SECRET_KEY` | `sk-lf-...` | Pasangan secret key |
| `LANGFUSE_BASE_URL` | `http://localhost:3000` | Alamat server Langfuse. Default-nya server lokal, bukan Langfuse Cloud |

Docker Compose memakai **pasangan key yang sama** untuk membuat project Langfuse secara otomatis.
Nilainya bebas (sebaiknya diawali `pk-lf-` / `sk-lf-`), tetapi jangan diubah setelah Langfuse pertama kali
dijalankan: inisialisasi otomatis hanya membuat project bila belum ada. Bila key harus diganti, buat key
baru di UI Langfuse (Project Settings > API Keys) lalu salin ke `.env`.

### Langfuse: server (Docker Compose)

| Variabel | Keterangan |
|---|---|
| `LANGFUSE_INIT_USER_EMAIL`, `LANGFUSE_INIT_USER_PASSWORD` | Akun admin untuk login ke UI, dibuat saat start pertama |
| `LANGFUSE_NEXTAUTH_SECRET`, `LANGFUSE_SALT` | Secret server. Buat dengan `openssl rand -base64 32` |
| `LANGFUSE_ENCRYPTION_KEY` | 64 karakter hex. Buat dengan `openssl rand -hex 32` |
| `POSTGRES_PASSWORD`, `CLICKHOUSE_PASSWORD`, `REDIS_AUTH`, `MINIO_ROOT_PASSWORD` | Password layanan internal Langfuse |
| `LANGFUSE_TELEMETRY_ENABLED` | Kirim statistik pemakaian anonim ke tim Langfuse (`true`/`false`) |

`docker compose` menolak berjalan bila `LANGFUSE_NEXTAUTH_SECRET`, `LANGFUSE_SALT`, atau
`LANGFUSE_ENCRYPTION_KEY` kosong. Ganti password layanan internal **sebelum** start pertama; setelah
volume data terbentuk, Postgres tidak membaca ulang password baru.

File `.env` harus berada di folder root project, sejajar dengan `src/` dan `compose.yaml`. Setelah
mengubah `.env`, restart server/CLI; `--reload` tidak memuat ulang file ini.

---

## 5. Menyiapkan data produk

Seed menerima file **CSV** atau **Excel** (`.xlsx`, `.xls`) dengan kolom berikut. Nama kolom tidak
membedakan huruf besar/kecil.

| Kolom | Wajib | Tipe | Keterangan |
|---|---|---|---|
| `id` | Ya | teks | ID unik produk, mis. `SKU001` |
| `name` | Ya | teks | Nama produk |
| `price` | Ya | angka | Harga dalam Rupiah, **angka polos** tanpa `Rp` dan tanpa titik pemisah. Harus lebih dari 0 |
| `description` | Tidak | teks | Deskripsi produk; makin informatif, makin baik hasil pencarian |
| `product_type` | Tidak | teks | Kategori, mis. `Sepatu`, `Elektronik` |
| `stock` | Tidak | bilangan bulat | Jumlah stok; default 0 |

Contoh isi file:

```csv
id,name,price,description,product_type,stock
SKU001,Nike Air Max 270,2199000,Sepatu lari ringan dengan bantalan Air Max,Sepatu,45
SKU005,Sony WH-1000XM5,4999000,Headphone wireless noise cancelling dengan baterai 30 jam,Elektronik,25
```

Muat file Anda dengan opsi `--file`:

```powershell
python -m src.db.seed --file data/produk_saya.xlsx
```

Hal yang perlu diketahui:

- Produk dengan `id` yang sama akan **ditimpa** dengan data baru.
- Produk yang sudah ada di database tetapi tidak ada di file baru **tidak dihapus**. Untuk mulai dari
  kosong, hentikan server, hapus `data/cs_agent.db` (beserta file `-wal` dan `-shm` bila ada), lalu seed ulang.
- Jika nama kolom di file Anda berbeda (mis. `nama_produk`, `harga`), ubah dulu header-nya agar sesuai tabel di atas.

---

## 6. Chat lewat command line

```powershell
python -m src.cli
```

Agent dijalankan langsung di terminal, tanpa perlu menyalakan server API. Selama agent bekerja, CLI
menampilkan **langkah-langkahnya secara langsung**, dari memahami pertanyaan sampai jawaban siap. Contoh sesi:

```
Customer Service Agent - chat CLI
mode lokal · pencarian: hybrid (BM25 + vector) · Langfuse: aktif
session: cli-1a2b3c4d · langkah: ringkas · ketik /help untuk daftar perintah, /exit untuk keluar

Anda  > Berapa harga Nike Air Max?
✓ Memahami pertanyaan → info produk spesifik (yakin 90%)  1.1 dtk
✓ Menentukan langkah → perlu data: cari katalog  0.9 dtk
✓ Mengambil data: cari katalog("Nike Air Max") → 5 produk, teratas Nike Air Max 270 (Rp 2.199.000)  40 ms
✓ Menyusun jawaban dari data produk  2.3 dtk
Agent > Nike Air Max 270 tersedia dengan harga Rp 2.199.000, stok 45.
  (intent: exact_fact · sumber: db_lookup · produk: SKU001 · 4.3 dtk)
  trace: http://localhost:3000/project/cs-agent/traces/...
```

(Kalimat jawaban dan durasi di atas hanya ilustrasi; nilainya bergantung pada model dan hardware.)

Riwayat percakapan disimpan selama sesi CLI berjalan, jadi pertanyaan lanjutan seperti
"kalau yang Adidas berapa?" dipahami dalam konteks.

### Tampilan langkah kerja agent

Yang ditampilkan adalah **langkah pipeline agent** (klasifikasi intent, pencarian produk, pemanggilan
tool, penyusunan jawaban), bukan teks reasoning internal model. Di terminal interaktif, langkah yang
sedang berjalan tampil dengan animasi dan timer, sehingga pengguna tahu agent sedang mengerjakan apa
dan berapa lama.

| Mode | Isi |
|---|---|
| `ringkas` (default) | Satu baris per langkah beserta hasil dan durasinya |
| `detail` | Ditambah kata kunci hasil ekstraksi, kandidat produk beserta skor pencarian, dan jumlah token output tiap panggilan LLM |
| `off` | Tidak menampilkan langkah |

Pilih mode saat start dengan `--steps detail`, atau ganti di tengah chat dengan `/steps` (berputar
ringkas → detail → off) atau `/steps detail`. Bila salah satu langkah gagal (mis. Ollama mati), langkah itu
ditandai `✗` beserta pesan error-nya.

Di console Windows lama yang tidak mendukung simbol Unicode, CLI otomatis memakai simbol ASCII.
Windows Terminal dan terminal VS Code menampilkan simbol penuh.

### Opsi

| Opsi | Keterangan |
|---|---|
| `--api URL` | Kirim pesan ke server API yang sedang berjalan, mis. `--api http://localhost:8000`, alih-alih menjalankan agent di terminal |
| `--session ID` | Set `session_id` (default dibuat acak). Semua giliran dengan ID sama dikelompokkan sebagai satu session di Langfuse |
| `--user ID` | Set `user_id` pelanggan, dicatat di Langfuse |
| `--steps MODE` | Tampilan langkah kerja agent: `ringkas` (default), `detail`, atau `off` |
| `--verbose`, `-v` | Tampilkan log teknis agent (output mentah classifier, dll). Hanya untuk mode lokal |
| `--no-meta` | Sembunyikan baris info intent/sumber/trace di bawah jawaban |
| `--no-color` | Matikan warna dan animasi (atau set environment variable `NO_COLOR`) |
| `--timeout DETIK` | Batas waktu request di mode `--api` (default 300) |

### Perintah di dalam chat

| Perintah | Fungsi |
|---|---|
| `/help` | Daftar perintah |
| `/steps [mode]` | Ganti tampilan langkah (`ringkas`, `detail`, `off`); tanpa argumen berputar ke mode berikutnya |
| `/reset` | Hapus riwayat dan mulai session baru |
| `/history` | Tampilkan riwayat percakapan |
| `/meta` | Tampilkan/sembunyikan info intent, sumber, dan link trace |
| `/exit` | Keluar (bisa juga Ctrl+C atau Ctrl+D) |

---

## 7. Menjalankan server API

```powershell
uvicorn src.main:app --reload
```

Log startup yang sehat terlihat seperti ini:

```
10:15:02 INFO    [cs_agent.startup] LLM base_url=http://localhost:11434/v1 model=qwen3.8:27b intent_model=qwen3.8:27b reasoning_effort=none
10:15:03 INFO    [cs_agent.startup] embedding 'bge-m3:567m' siap (dimensi 1024)
10:15:03 INFO    [cs_agent.search] BM25: 8 produk ter-index
10:15:03 INFO    [cs_agent.startup] Langfuse tracing aktif -> http://localhost:3000
10:15:03 INFO    [cs_agent.startup] ready.
```

Cara membaca log tersebut:

- Baris `embedding ... siap` berarti hybrid search aktif penuh.
- Jika muncul `pakai BM25 saja`, server tetap berjalan tetapi hanya dengan pencarian kata kunci.
  Penyebabnya dijelaskan di [Troubleshooting](#13-troubleshooting).
- Jika `tidak ada produk, index BM25 kosong`, database masih kosong: jalankan seed, lalu restart server.
- Baris `Langfuse tracing ...` menunjukkan status observability.

Untuk diakses dari perangkat lain di jaringan yang sama, jalankan tanpa `--reload` dan buka host-nya:

```powershell
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

---

## 8. Memakai API

Dokumentasi interaktif tersedia di http://localhost:8000/docs. Dari halaman itu Anda bisa mencoba
endpoint langsung lewat browser.

### `GET /api/v1/health`

Mengecek apakah server hidup. Contoh balasan:

```json
{"status": "ok", "vector_search": true, "tracing": true}
```

### `POST /api/v1/chat`

Body request:

| Field | Wajib | Keterangan |
|---|---|---|
| `message` | Ya | Pesan pelanggan, 1 sampai 4000 karakter |
| `session_id` | Tidak | ID percakapan. Default `"default"`. Di Langfuse, giliran dengan ID sama dikelompokkan menjadi satu session |
| `user_id` | Tidak | ID pelanggan (maks. 200 karakter), dicatat di Langfuse |
| `history` | Tidak | Riwayat percakapan sebelumnya, maksimal 20 pesan. Tiap pesan berisi `role` (`user` atau `assistant`) dan `content` |

Contoh dengan `curl` (Linux/macOS/Git Bash):

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Berapa harga Nike Air Max?", "session_id": "web-123", "history": []}'
```

Contoh dengan PowerShell:

```powershell
$body = @{ message = "Berapa harga Nike Air Max?"; session_id = "web-123"; history = @() } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/api/v1/chat -Method Post -ContentType "application/json" -Body $body
```

Contoh response:

```json
{
  "reply": "Nike Air Max 270 tersedia dengan harga Rp 2.199.000, stok 45.",
  "intent": "exact_fact",
  "products_shown": ["SKU001"],
  "sources": ["db_lookup"],
  "session_id": "web-123",
  "trace_url": "http://localhost:3000/project/cs-agent/traces/..."
}
```

| Field response | Keterangan |
|---|---|
| `reply` | Jawaban untuk pelanggan |
| `intent` | Hasil klasifikasi: `general`, `product_search`, atau `exact_fact` |
| `products_shown` | ID produk yang diambil dari database pada giliran ini, termasuk kandidat hasil pencarian yang belum tentu disebut di jawaban |
| `sources` | Asal jawaban: `llm_only`, `hybrid_search`, `db_lookup`, atau `error` |
| `session_id` | Sama dengan yang dikirim |
| `trace_url` | Link trace giliran ini di Langfuse; `null` bila tracing nonaktif |

### Percakapan lanjutan (multi-turn)

Server **tidak menyimpan** riwayat percakapan. Aplikasi pemanggil yang harus mengirim ulang riwayat
di field `history` pada setiap request. Jangan mengirim `history` berisi teks contoh seperti
`"string"` dari halaman `/docs`, karena itu ikut dibaca model.

```json
{
  "message": "Kalau yang Adidas berapa?",
  "session_id": "web-123",
  "history": [
    {"role": "user", "content": "Berapa harga Nike Air Max?"},
    {"role": "assistant", "content": "Nike Air Max 270 tersedia dengan harga Rp 2.199.000, stok 45."}
  ]
}
```

Contoh klien sederhana sudah tersedia: `python -m src.cli --api http://localhost:8000`.

### `POST /api/v1/chat/stream`

Body request sama dengan `/chat`, tetapi response dikirim sebagai **Server-Sent Events** sehingga
frontend bisa menampilkan langkah kerja agent secara langsung (seperti di CLI):

```
event: step
data: {"type": "intent.done", "data": {"intent": "exact_fact", "confidence": 0.9, "query": "Nike Air Max", "duration": 1.1}, "ts": 1760086800.1}

event: step
data: {"type": "tool.start", "data": {"name": "search_product_catalog", "args": {"query": "Nike Air Max"}}, "ts": ...}

...

event: result
data: {"reply": "...", "intent": "exact_fact", "products_shown": [...], "sources": ["db_lookup"], "session_id": "...", "trace_url": "..."}
```

| Event | Kapan | Isi |
|---|---|---|
| `step` | Berkali-kali selama agent bekerja | `type` berformat `<langkah>.start`, `<langkah>.done`, atau `<langkah>.error`; langkah: `turn`, `intent`, `llm`, `tool`, `search` |
| `result` | Sekali, di akhir | Sama persis dengan response `/chat` |
| `error` | Bila agent gagal | `{"detail": "..."}` |

Field `data` tiap langkah:

| Langkah | Field |
|---|---|
| `intent` | `intent`, `confidence`, `query`, `product_id`, `fallback` (alasan bila classifier gagal) |
| `llm` | `purpose` (`answer` / `tool_round`), `round`, `next` (`tools` / `answer`), `tools`, `output_tokens` |
| `tool` | `name`, `args`, `found`, `count`, `top` (maks. 3 produk: `id`, `name`, `price`, `stock`), `error` |
| `search` | `query`, `method`, `count`, `top` (maks. 3 produk: `id`, `name`, `price`, `score`) |
| `turn` | `message`, lalu `intent`, `sources`, `products` saat selesai |

Semua event `.done` dan `.error` menyertakan `duration` (detik).

Contoh dengan `curl` (`-N` agar output tidak di-buffer):

```bash
curl -N -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Berapa harga Nike Air Max?"}'
```

---

## 9. Observability dengan Langfuse

Saat tracing aktif, **setiap giliran chat menjadi satu trace** di Langfuse, baik dari CLI maupun API.

### Yang tercatat

```
customer-service-turn                (agent)     input pesan, output jawaban, intent, sumber
├── intent-classification            (chain)     hasil intent + confidence
│   └── intent-classifier            (generation) prompt, output JSON mentah, model, token, latensi
├── lane-exact-fact                  (chain)
│   ├── tool-calling-round-1         (generation) LLM memutuskan memanggil tool
│   ├── tool:search_product_catalog  (tool)       argumen dan hasil query database
│   │   └── hybrid-search            (retriever)  query, produk + skor, metode (hybrid/BM25)
│   │       └── OpenAI-embedding     (embedding)  embedding query
│   └── tool-calling-round-2         (generation) LLM menyusun jawaban dari hasil tool
```

Jalur `product_search` mencatat `hybrid-search` lalu generation `answer-product-search`; jalur `general`
mencatat generation `answer-general`.

Atribut setiap trace:

- **Session**: dari `session_id` (CLI: dibuat otomatis per sesi). Halaman *Sessions* menampilkan satu
  percakapan utuh dari awal sampai akhir.
- **User**: dari `user_id` (API) atau `--user` (CLI).
- **Tag**: `intent:general`, `intent:product_search`, atau `intent:exact_fact`, untuk memfilter trace per jalur.
- **Metadata**: nama model jawaban dan model intent.

### Cara memakai

1. Pastikan Langfuse berjalan (`docker compose ps`) dan key di `.env` terisi.
2. Jalankan CLI atau server; log startup menampilkan `Langfuse tracing aktif`.
3. Kirim beberapa pesan, lalu buka http://localhost:3000 > project **Customer Service Agent** > **Tracing**.
   CLI dan response API juga menyertakan link langsung ke trace (`trace:` / `trace_url`).

Hal yang berguna dipantau:

- Trace dengan level `WARNING` atau `ERROR`: classifier gagal menghasilkan JSON, atau loop tool melebihi batas.
- Latensi per generation, untuk melihat langkah mana yang paling lambat.
- Pemakaian token per model, dari dashboard Langfuse.

### Perilaku saat Langfuse tidak tersedia

Tracing tidak pernah menghentikan agent. Bila server Langfuse mati atau key salah, agent tetap menjawab,
log startup menampilkan peringatan, dan trace yang gagal dikirim dibuang setelah beberapa kali percobaan.
Untuk mematikan tracing sepenuhnya, set `LANGFUSE_ENABLED=false`; library `langfuse` tidak di-import sama sekali.

---

## 10. Mengganti data atau model

| Yang diubah | Yang harus dilakukan |
|---|---|
| Data produk (harga, stok, produk baru) | Jalankan seed ulang, lalu **restart server/CLI** agar index BM25 dibangun ulang |
| `EMBEDDING_MODEL` | Jalankan seed ulang (tabel vektor dibuat ulang mengikuti dimensi model), lalu restart |
| `OPENAI_MODEL` / `OPENAI_MODEL_FAST` | Cukup restart server/CLI |
| System prompt / aturan jawaban | Ubah di `src/agent/core.py` dan `src/agent/intent.py`; `--reload` memuatnya otomatis |

---

## 11. Memakai OpenAI sebagai pengganti Ollama

Ubah `.env` menjadi:

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_MODEL_FAST=gpt-4o-mini
LLM_REASONING_EFFORT=
EMBEDDING_MODEL=text-embedding-3-small
```

Hapus baris `OPENAI_BASE_URL`, lalu jalankan seed ulang (dimensi embedding berbeda) dan restart server.
Gunakan model chat non-reasoning; kode mengirim parameter `max_tokens` dan `temperature`.
Langfuse tetap bisa dipakai tanpa perubahan.

---

## 12. Struktur project

```
customer_service_agent/
├── .env.example               # template konfigurasi (aplikasi + Docker Compose)
├── compose.yaml               # Ollama + Langfuse (web, worker, postgres, clickhouse, redis, minio)
├── requirements.txt
├── data/
│   ├── products.csv           # data contoh
│   └── cs_agent.db            # dibuat otomatis oleh seed
└── src/
    ├── config.py              # membaca .env
    ├── bootstrap.py           # inisialisasi bersama untuk server API dan CLI
    ├── observability.py       # integrasi Langfuse (no-op bila nonaktif)
    ├── main.py                # aplikasi FastAPI
    ├── cli.py                 # chat lewat command line
    ├── cli_steps.py           # tampilan langkah kerja agent di CLI
    ├── api/routes.py          # endpoint /health, /chat, /chat/stream
    ├── agent/
    │   ├── events.py          # event langkah kerja agent (untuk CLI dan streaming)
    │   ├── intent.py          # klasifikasi intent
    │   └── core.py            # tiga jalur jawaban + loop tool calling
    ├── tools/product_tools.py # tool yang bisa dipanggil LLM
    ├── retrieval/
    │   ├── embedder.py        # embedding lewat Ollama
    │   └── hybrid_search.py   # BM25 + vector + RRF
    ├── db/
    │   ├── database.py        # koneksi, schema, query
    │   └── seed.py            # memuat CSV/Excel ke database
    └── models/schemas.py      # model data + format Rupiah
```

Tool yang tersedia untuk LLM di jalur `exact_fact`:

| Tool | Fungsi |
|---|---|
| `search_product_catalog(query)` | Mencari produk berdasarkan nama, tipe, atau deskripsi |
| `get_product_price(product_id)` | Mengambil harga dan stok berdasarkan ID |
| `get_product_detail(product_id)` | Mengambil detail lengkap berdasarkan ID |

---

## 13. Troubleshooting

### Agent

| Gejala | Penyebab dan solusi |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Perintah dijalankan dari folder yang salah. Pindah ke folder root project (yang berisi `src/`) |
| `RuntimeError: OPENAI_API_KEY belum diset` | File `.env` belum ada, salah lokasi, atau tersimpan sebagai `.env.txt`. Cek dengan `dir /a` |
| `Connection error` / `Connection refused` saat chat | Ollama tidak berjalan atau `OPENAI_BASE_URL` salah. Buka http://localhost:11434 untuk memastikan |
| `model '...' not found` | Nama model di `.env` tidak sama dengan `ollama list`, atau model belum di-pull (pada Ollama Docker: `docker exec -it ollama ollama list`) |
| `reply` kosong atau berisi pesan maaf, log `[cs_agent.intent] ... raw=''` | Thinking mode belum mati. Pastikan `LLM_REASONING_EFFORT=none`, update Ollama ke versi terbaru, atau matikan thinking lewat Modelfile |
| Semua pesan diklasifikasikan `general` | Classifier gagal menghasilkan JSON. Jalankan `python -m src.cli -v` atau lihat generation `intent-classifier` di Langfuse untuk output mentahnya |
| `... does not support tools` | Model tidak mendukung tool calling. Ganti ke model yang mendukungnya |
| Jawaban tidak lengkap atau mengabaikan sebagian produk | Context window terlalu kecil. Naikkan `OLLAMA_CONTEXT_LENGTH` lalu restart Ollama |
| Log `sqlite-vec tidak tersedia` | Ekstensi vektor gagal dimuat di instalasi Python ini. Agent tetap berjalan dengan BM25 saja |
| Log `index vektor belum ada` | Seed belum dijalankan, atau embedding gagal saat seed. Jalankan seed ulang dan baca pesannya |
| Log `[seed] embedding dilewati: ...` | Ollama mati atau model embedding belum di-pull saat seed dijalankan |
| Log `vector_search gagal` (dimensi tidak cocok) | `EMBEDDING_MODEL` diganti tanpa seed ulang. Jalankan seed ulang |
| Data baru tidak muncul di jawaban | Server/CLI belum di-restart setelah seed |
| Respons sangat lambat | Tiap pesan memanggil LLM minimal dua kali. Pakai model lebih kecil untuk `OPENAI_MODEL_FAST` |
| CLI mode `--api`: `Server API ... tidak bisa dihubungi` | Server belum dijalankan (`uvicorn src.main:app`) atau URL/port salah |
| CLI menampilkan kotak atau karakter aneh, bukan simbol ✓ dan animasi | Console tidak mendukung Unicode/ANSI. Pakai Windows Terminal, atau jalankan dengan `--no-color` |
| Langkah tidak muncul di mode `--api` | Server versi lama belum punya `/api/v1/chat/stream`; CLI otomatis kembali ke `/chat` tanpa langkah. Perbarui server |

### Langfuse dan Docker

| Gejala | Penyebab dan solusi |
|---|---|
| `required variable LANGFUSE_SALT is missing a value` | Secret server di `.env` kosong. Isi `LANGFUSE_NEXTAUTH_SECRET`, `LANGFUSE_SALT`, `LANGFUSE_ENCRYPTION_KEY` |
| `port is already allocated` (11434) | Ollama di komputer sedang berjalan. Pakai `docker compose up -d langfuse-web langfuse-worker` |
| `port is already allocated` (3000 atau 9090) | Aplikasi lain memakai port itu. Hentikan aplikasinya atau ubah port di sisi kiri `ports` pada `compose.yaml` (lalu sesuaikan `LANGFUSE_BASE_URL`) |
| http://localhost:3000 belum bisa dibuka | Langfuse masih migrasi database saat start pertama. Tunggu, lalu cek `docker compose logs -f langfuse-web` |
| Log `Langfuse tracing aktif tetapi server/key belum valid` | Server belum siap, `LANGFUSE_BASE_URL` salah, atau key di `.env` berbeda dari project. Bila key diubah setelah start pertama, buat key baru di UI dan salin ke `.env` |
| Tidak bisa login ke UI | Akun dibuat dari `LANGFUSE_INIT_USER_*` hanya saat start pertama. Bila nilainya diubah setelahnya, akun baru tidak dibuat |
| Log `Transient error ... exporting spans` | Trace gagal dikirim karena Langfuse tidak bisa dihubungi. Agent tetap berjalan; nyalakan Langfuse atau set `LANGFUSE_ENABLED=false` |
| Trace tidak muncul padahal tracing aktif | Pada server API, trace dikirim berkala (beberapa detik). Refresh halaman Tracing; trace yang tersisa dikirim saat server dihentikan |
| GPU tidak terdeteksi oleh container `ollama` | Bagian `deploy.resources` butuh driver NVIDIA dan dukungan GPU di Docker Desktop. Tanpa GPU NVIDIA, hapus bagian `deploy` dari layanan `ollama` |

Untuk mengulang Langfuse dari awal (**semua trace terhapus**): `docker compose down -v`, lalu
`docker compose up -d`. Perintah ini juga menghapus model yang tersimpan di volume Ollama Docker.

---

## 14. Batasan

- **Tidak ada autentikasi** di API dan CORS dibuka untuk semua origin. Jangan diekspos ke internet tanpa
  menambahkan pengaman.
- **Konfigurasi Langfuse ditujukan untuk satu mesin lokal.** Untuk server bersama, ganti semua secret,
  pasang HTTPS, dan ikuti panduan self-hosting resmi Langfuse.
- **Isi percakapan pelanggan tersimpan di Langfuse** (input, output, prompt). Pertimbangkan kebijakan
  privasi sebelum dipakai dengan data pelanggan sungguhan.
- **Riwayat percakapan tidak disimpan di server API**; pengelolaannya menjadi tanggung jawab aplikasi pemanggil.
- **Kualitas jawaban bergantung pada model.** Arsitektur menjamin angka berasal dari database, tetapi
  model yang lemah masih bisa salah memilih produk atau salah menyalin. Uji dengan pertanyaan nyata
  sebelum dipakai pelanggan.
- Pengetahuan agent terbatas pada data produk. Pertanyaan tentang kebijakan toko (retur, ongkir)
  dijawab secara umum karena datanya belum tersedia.
- SQLite cocok untuk katalog kecil sampai menengah dengan satu instance server.