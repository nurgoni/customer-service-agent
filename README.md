# Customer Service Agent

Chatbot customer service yang menjawab pertanyaan pelanggan tentang produk (nama, harga, stok, deskripsi)
berdasarkan data produk Anda sendiri. LLM dijalankan lokal lewat **Ollama**, data disimpan di **SQLite**,
dan agent diakses lewat **REST API (FastAPI)**.

Prinsip utamanya: **harga dan stok tidak pernah dikarang oleh LLM**. Angka selalu diambil dari database
dan diformat oleh kode; LLM hanya merangkai kalimat jawabannya.

## Daftar isi

1. [Cara kerja](#1-cara-kerja)
2. [Prasyarat](#2-prasyarat)
3. [Instalasi dari nol](#3-instalasi-dari-nol)
4. [Konfigurasi `.env`](#4-konfigurasi-env)
5. [Menyiapkan data produk](#5-menyiapkan-data-produk)
6. [Menjalankan server](#6-menjalankan-server)
7. [Memakai API](#7-memakai-api)
8. [Mengganti data atau model](#8-mengganti-data-atau-model)
9. [Memakai OpenAI sebagai pengganti Ollama](#9-memakai-openai-sebagai-pengganti-ollama)
10. [Struktur project](#10-struktur-project)
11. [Troubleshooting](#11-troubleshooting)
12. [Batasan](#12-batasan)

---

## 1. Cara kerja

Setiap pesan pelanggan melewati dua tahap: klasifikasi intent, lalu salah satu dari tiga jalur.

```
Pesan pelanggan
      |
      v
Intent classifier (LLM, output JSON)
      |
      +-- general ---------> LLM menjawab langsung (sapaan, cara order, dll)
      |
      +-- product_search --> Hybrid search (BM25 + vector) -> LLM merangkum hasil
      |
      +-- exact_fact ------> LLM memanggil tool -> query SQLite -> LLM menyusun jawaban
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
| Ollama | Unduh dari https://ollama.com/download lalu pastikan `ollama --version` berjalan |
| Model LLM | Model yang mendukung *tool calling*, mis. keluarga Qwen. Default project: `qwen3.8:27b` |
| Model embedding | `bge-m3:567m` |
| Hardware | Model 27B butuh GPU/RAM besar. Jika mesin terbatas, pakai varian yang lebih kecil dan sesuaikan `.env` |

Project ini dikembangkan di Windows. Perintah untuk Linux/macOS disertakan di tiap langkah.

---

## 3. Instalasi dari nol

Semua perintah dijalankan dari **folder root project**, yaitu folder yang berisi `src/` dan `requirements.txt`.

### Langkah 1 - Ambil kode

Ekstrak zip project (atau clone repository), lalu masuk ke foldernya:

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

### Langkah 4 - Siapkan Ollama

Unduh kedua model. Ganti nama model LLM bila Anda memakai varian lain.

```powershell
ollama pull qwen3.8:27b
ollama pull bge-m3:567m
ollama list
```

Catat nama model persis seperti yang tampil di `ollama list`; nama itu yang diisi di `.env`.

Naikkan context window Ollama. Nilai default-nya kecil, sehingga prompt berisi daftar produk bisa
terpotong tanpa peringatan dan jawaban jadi tidak akurat.

Windows:

```powershell
setx OLLAMA_CONTEXT_LENGTH 8192
```

Setelah itu **tutup dan jalankan ulang Ollama** (keluar dari ikon tray, lalu buka lagi).

Linux/macOS:

```bash
OLLAMA_CONTEXT_LENGTH=8192 ollama serve
```

Pastikan Ollama aktif dengan membuka http://localhost:11434 di browser. Halaman itu harus menampilkan
`Ollama is running`.

### Langkah 5 - Buat file `.env`

Windows:

```powershell
copy .env.example .env
```

Linux/macOS:

```bash
cp .env.example .env
```

Buka `.env` dan sesuaikan nama model dengan hasil `ollama list`. Penjelasan tiap variabel ada di
[bagian 4](#4-konfigurasi-env).

### Langkah 6 - Muat data produk ke database

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

### Langkah 7 - Jalankan server

```powershell
uvicorn src.main:app --reload
```

Server siap ketika muncul `[ startup ] ready.`. Lanjut ke [bagian 7](#7-memakai-api) untuk mencobanya.

---

## 4. Konfigurasi `.env`

| Variabel | Contoh | Keterangan |
|---|---|---|
| `OPENAI_API_KEY` | `ollama` | Wajib diisi. Ollama tidak memeriksa nilainya, jadi isi bebas asal tidak kosong |
| `OPENAI_BASE_URL` | `http://localhost:11434/v1` | Alamat Ollama. Akhiran `/v1` wajib ada |
| `OPENAI_MODEL` | `qwen3.8:27b` | Model untuk menyusun jawaban dan memanggil tool |
| `OPENAI_MODEL_FAST` | `qwen3.8:27b` | Model untuk intent classifier. Boleh model lebih kecil agar lebih cepat |
| `LLM_REASONING_EFFORT` | `none` | Mematikan *thinking mode* pada model seperti Qwen3. Kosongkan untuk tidak mengirim parameter ini |
| `EMBEDDING_MODEL` | `bge-m3:567m` | Model embedding di Ollama |
| `EMBEDDING_BASE_URL` | (kosong) | Opsional. Isi hanya bila embedding dilayani server lain; default-nya sama dengan `OPENAI_BASE_URL` |

Nama variabel memakai awalan `OPENAI_` karena project ini berkomunikasi dengan Ollama lewat
library `openai` (Ollama menyediakan endpoint yang kompatibel).

File `.env` harus berada di folder root project, sejajar dengan `src/`. Setelah mengubah `.env`,
server harus di-restart; `--reload` tidak memuat ulang file ini.

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

## 6. Menjalankan server

```powershell
uvicorn src.main:app --reload
```

Log startup yang sehat terlihat seperti ini:

```
[ startup ] LLM base_url=http://localhost:11434/v1 model=qwen3.8:27b intent_model=qwen3.8:27b reasoning_effort=none
[ startup ] connecting to database...
[ startup ] checking embedding model...
[ startup ] embedding 'bge-m3:567m' siap (dimensi 1024)
[ startup ] building BM25 index...
[ bm25 ] 8 produk ter-index
[ startup ] initialising agent...
[ startup ] ready.
```

Cara membaca log tersebut:

- Baris `embedding ... siap` berarti hybrid search aktif penuh.
- Jika muncul `pakai BM25 saja`, server tetap berjalan tetapi hanya dengan pencarian kata kunci.
  Penyebabnya dijelaskan di [Troubleshooting](#11-troubleshooting).
- Jika `[ bm25 ] tidak ada produk`, database masih kosong: jalankan seed, lalu restart server.

Untuk diakses dari perangkat lain di jaringan yang sama, jalankan tanpa `--reload` dan buka host-nya:

```powershell
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

---

## 7. Memakai API

Dokumentasi interaktif tersedia di http://localhost:8000/docs. Dari halaman itu Anda bisa mencoba
endpoint langsung lewat browser.

### `GET /api/v1/health`

Mengecek apakah server hidup. Balasannya `{"status": "ok"}`.

### `POST /api/v1/chat`

Body request:

| Field | Wajib | Keterangan |
|---|---|---|
| `message` | Ya | Pesan pelanggan, 1 sampai 4000 karakter |
| `session_id` | Tidak | Label sesi; dikembalikan apa adanya di response. Default `"default"` |
| `history` | Tidak | Riwayat percakapan sebelumnya, maksimal 20 pesan. Tiap pesan berisi `role` (`user` atau `assistant`) dan `content` |

Contoh dengan `curl` (Linux/macOS/Git Bash):

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Berapa harga Nike Air Max?", "history": []}'
```

Contoh dengan PowerShell:

```powershell
$body = @{ message = "Berapa harga Nike Air Max?"; history = @() } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/api/v1/chat -Method Post -ContentType "application/json" -Body $body
```

Contoh response:

```json
{
  "reply": "Nike Air Max 270 tersedia dengan harga Rp 2.199.000, stok 45.",
  "intent": "exact_fact",
  "products_shown": ["SKU001"],
  "sources": ["db_lookup"],
  "session_id": "default"
}
```

| Field response | Keterangan |
|---|---|
| `reply` | Jawaban untuk pelanggan |
| `intent` | Hasil klasifikasi: `general`, `product_search`, atau `exact_fact` |
| `products_shown` | ID produk yang diambil dari database pada giliran ini |
| `sources` | Asal jawaban: `llm_only`, `hybrid_search`, `db_lookup`, atau `error` |
| `session_id` | Sama dengan yang dikirim |

### Percakapan lanjutan (multi-turn)

Server **tidak menyimpan** riwayat percakapan. Aplikasi pemanggil yang harus mengirim ulang riwayat
di field `history` pada setiap request. Jangan mengirim `history` berisi teks contoh seperti
`"string"` dari halaman `/docs`, karena itu ikut dibaca model.

```json
{
  "message": "Kalau yang Adidas berapa?",
  "history": [
    {"role": "user", "content": "Berapa harga Nike Air Max?"},
    {"role": "assistant", "content": "Nike Air Max 270 tersedia dengan harga Rp 2.199.000, stok 45."}
  ]
}
```

Contoh pemanggilan dari Python yang menjaga riwayat:

```python
import requests

URL = "http://localhost:8000/api/v1/chat"
history = []

while True:
    message = input("Anda: ")
    if message.lower() in ("quit", "exit"):
        break
    data = requests.post(URL, json={"message": message, "history": history[-20:]}).json()
    print("Agent:", data["reply"])
    history += [
        {"role": "user", "content": message},
        {"role": "assistant", "content": data["reply"]},
    ]
```

(Jalankan `pip install requests` bila belum terpasang.)

---

## 8. Mengganti data atau model

| Yang diubah | Yang harus dilakukan |
|---|---|
| Data produk (harga, stok, produk baru) | Jalankan seed ulang, lalu **restart server** agar index BM25 dibangun ulang |
| `EMBEDDING_MODEL` | Jalankan seed ulang (tabel vektor dibuat ulang mengikuti dimensi model), lalu restart server |
| `OPENAI_MODEL` / `OPENAI_MODEL_FAST` | Cukup restart server |
| System prompt / aturan jawaban | Ubah di `src/agent/core.py` dan `src/agent/intent.py`; `--reload` memuatnya otomatis |

---

## 9. Memakai OpenAI sebagai pengganti Ollama

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

---

## 10. Struktur project

```
customer_service_agent/
├── .env.example               # template konfigurasi
├── requirements.txt
├── data/
│   ├── products.csv           # data contoh
│   └── cs_agent.db            # dibuat otomatis oleh seed
└── src/
    ├── config.py              # membaca .env
    ├── main.py                # aplikasi FastAPI + proses startup
    ├── api/routes.py          # endpoint /health dan /chat
    ├── agent/
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

## 11. Troubleshooting

| Gejala | Penyebab dan solusi |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Perintah dijalankan dari folder yang salah. Pindah ke folder root project (yang berisi `src/`) |
| `RuntimeError: OPENAI_API_KEY belum diset` | File `.env` belum ada, salah lokasi, atau tersimpan sebagai `.env.txt`. Cek dengan `dir /a` |
| `Connection error` / `Connection refused` saat chat | Ollama tidak berjalan atau `OPENAI_BASE_URL` salah. Buka http://localhost:11434 untuk memastikan |
| `model '...' not found` | Nama model di `.env` tidak sama dengan `ollama list`, atau model belum di-pull |
| `reply` kosong atau berisi pesan maaf, log `[intent] ... raw=''` | Thinking mode belum mati. Pastikan `LLM_REASONING_EFFORT=none`, update Ollama ke versi terbaru, atau matikan thinking lewat Modelfile |
| Semua pesan diklasifikasikan `general` | Classifier gagal menghasilkan JSON. Lihat baris log `[intent]` untuk output mentahnya; biasanya sama dengan masalah thinking mode di atas |
| `... does not support tools` | Model tidak mendukung tool calling. Ganti ke model yang mendukungnya |
| Jawaban tidak lengkap atau mengabaikan sebagian produk | Context window terlalu kecil. Set `OLLAMA_CONTEXT_LENGTH=8192` lalu restart Ollama |
| Log `sqlite-vec tidak tersedia` | Ekstensi vektor gagal dimuat di instalasi Python ini. Server tetap berjalan dengan BM25 saja |
| Log `index vektor belum ada` | Seed belum dijalankan, atau embedding gagal saat seed. Jalankan seed ulang dan baca pesannya |
| Log `[seed] embedding dilewati: ...` | Ollama mati atau model embedding belum di-pull saat seed dijalankan |
| Log `vector_search gagal` (dimensi tidak cocok) | `EMBEDDING_MODEL` diganti tanpa seed ulang. Jalankan seed ulang |
| Data baru tidak muncul di jawaban | Server belum di-restart setelah seed |
| Respons sangat lambat | Tiap pesan memanggil LLM minimal dua kali. Pakai model lebih kecil untuk `OPENAI_MODEL_FAST` |

Log yang membantu saat debugging:

- `[intent] model=... finish_reason=... raw=...` menampilkan output mentah classifier.
- `[agent] tool=... args=...` menampilkan tool yang dipanggil model beserta argumennya.

---

## 12. Batasan

- **Tidak ada autentikasi** dan CORS dibuka untuk semua origin. Jangan diekspos ke internet tanpa
  menambahkan pengaman.
- **Riwayat percakapan tidak disimpan di server**; pengelolaannya menjadi tanggung jawab aplikasi pemanggil.
- **Kualitas jawaban bergantung pada model.** Arsitektur menjamin angka berasal dari database, tetapi
  model yang lemah masih bisa salah memilih produk atau salah menyalin. Uji dengan pertanyaan nyata
  sebelum dipakai pelanggan.
- Pengetahuan agent terbatas pada data produk. Pertanyaan tentang kebijakan toko (retur, ongkir)
  dijawab secara umum karena datanya belum tersedia.
- SQLite cocok untuk katalog kecil sampai menengah dengan satu instance server.