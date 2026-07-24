"""
Konfigurasi aplikasi — SEMUA nilai sensitif/lingkungan dibaca dari
environment variable, tidak pernah di-hardcode di kode.

Tahap 1 & 2 memakai variabel OCR/ekstraksi & database lokal. Variabel
Google Sheets/Drive sudah didaftarkan di sini agar konsisten dengan skema
penuh proyek, tapi belum dipakai sampai Tahap 4.
"""
import os
from pathlib import Path


class ConfigError(RuntimeError):
    """Dilempar jika environment variable wajib untuk tahap ini tidak ada."""


# --- Wajib untuk Tahap 1 (OCR) ---
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OCR_MODEL_NAME = os.environ.get("OCR_MODEL_NAME", "glm-ocr")

# Context window (token) yang diminta ke Ollama saat OCR. Default Ollama
# (4096) sering terlampaui oleh gambar hasil scan dokumen beresolusi
# tinggi + prompt. Default 8192 di sini sudah terbukti cukup dan aman
# untuk GPU 4GB VRAM (mis. RTX 3050). Naikkan lewat env var OCR_NUM_CTX
# jika masih terjadi error "exceeds the available context size"
# (untuk GPU dengan VRAM lebih besar), atau turunkan jika terjadi
# "CUDA error: out of memory" (untuk GPU lebih kecil).
OCR_NUM_CTX = int(os.environ.get("OCR_NUM_CTX", "8192"))

# --- Wajib untuk Tahap 2 (ekstraksi field terstruktur) ---
EXTRACTION_MODEL_NAME = os.environ.get("EXTRACTION_MODEL_NAME", "qwen2.5:7b")

# Context window untuk model ekstraksi. Perlu lebih besar dari OCR karena
# input berupa gabungan teks OCR semua halaman + prompt instruksi +
# few-shot example, bisa cukup panjang untuk surat multi-halaman.
EXTRACTION_NUM_CTX = int(os.environ.get("EXTRACTION_NUM_CTX", "8192"))

# Batasi ekstraksi hanya ke N halaman pertama dokumen (mengecualikan
# lampiran di halaman berikutnya). Berdasarkan pengamatan proses bisnis:
# isi surat resmi (nomor surat, tanggal, penandatangan, dst) selalu ada
# di 1-2 halaman pertama, sedangkan lampiran di halaman setelahnya kurang
# diverifikasi ketat oleh pembuat/validator sehingga rawan jadi sumber
# salah ekstraksi jika ikut dibaca model. Set ke kosong/0 untuk
# menonaktifkan (baca semua halaman lagi).
_max_pages_raw = os.environ.get("EXTRACTION_MAX_PAGES", "2")
EXTRACTION_MAX_PAGES = int(_max_pages_raw) if _max_pages_raw.strip() else None
if EXTRACTION_MAX_PAGES == 0:
    EXTRACTION_MAX_PAGES = None

# --- Path database lokal (SQLite) ---
# Default: ~/.local/share/glm-ocr-app/app.db (bisa dioverride via env var)
_DEFAULT_DB_DIR = Path.home() / ".local" / "share" / "glm-ocr-app"
DB_PATH = Path(os.environ.get("GLM_OCR_DB_PATH", str(_DEFAULT_DB_DIR / "app.db")))

# --- Belum dipakai sampai Tahap 4 (Sheets & Drive) ---
GOOGLE_APPLICATION_CREDENTIALS = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
GOOGLE_SHEET_ID = os.environ.get("GOOGLE_SHEET_ID")
GOOGLE_DRIVE_FOLDER_ID = os.environ.get("GOOGLE_DRIVE_FOLDER_ID")


def ensure_db_dir() -> None:
    """Pastikan folder tujuan file database sudah ada."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
