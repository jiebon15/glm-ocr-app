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

# Ekstraksi dilakukan via DUA panggilan model terpisah (lihat app/extraction.py):
#   1. Field administratif (nomor surat, tanggal, dst) -> dari halaman 1-2 saja.
#   2. Field lain (petugas, UPI, no_reg, dst) -> dari SELURUH halaman dokumen
#      termasuk lampiran, karena field ini bisa muncul di halaman mana pun.

# Context window untuk panggilan #1 (field administratif, teks pendek
# karena dibatasi EXTRACTION_MAX_PAGES).
EXTRACTION_NUM_CTX = int(os.environ.get("EXTRACTION_NUM_CTX", "8192"))

# Context window untuk panggilan #2 (field lain, teks SELURUH dokumen
# termasuk lampiran -> butuh context lebih besar untuk dokumen panjang).
EXTRACTION_FULL_NUM_CTX = int(os.environ.get("EXTRACTION_FULL_NUM_CTX", "16384"))

# Batasi teks yang dipakai KHUSUS untuk panggilan #1 (field administratif)
# ke N halaman pertama dokumen. Berdasarkan pengamatan proses bisnis: isi
# inti surat (nomor surat, tanggal, penandatangan, dst) normalnya 1
# halaman, kadang terpotong ke halaman 2. Field lain (petugas, UPI, dst)
# TIDAK dibatasi ini — selalu dibaca dari seluruh dokumen.
_max_pages_raw = os.environ.get("EXTRACTION_MAX_PAGES", "2")
EXTRACTION_MAX_PAGES = int(_max_pages_raw) if _max_pages_raw.strip() else None
if EXTRACTION_MAX_PAGES == 0:
    EXTRACTION_MAX_PAGES = None

# --- Path database lokal (SQLite) ---
# Default: ~/.local/share/glm-ocr-app/app.db (bisa dioverride via env var)
_DEFAULT_DB_DIR = Path.home() / ".local" / "share" / "glm-ocr-app"
DB_PATH = Path(os.environ.get("GLM_OCR_DB_PATH", str(_DEFAULT_DB_DIR / "app.db")))

# --- Wajib untuk Tahap 3 (GUI) ---
# Folder penyimpanan file PDF yang di-upload lewat GUI Streamlit. File di
# sini bersifat SEMENTARA — akan dihapus otomatis setelah berhasil
# di-upload ke Google Drive (Tahap 4), sesuai desain "sumber kebenaran
# ada di Drive, tidak disimpan ganda secara lokal".
PDF_STORAGE_DIR = Path(
    os.environ.get("PDF_STORAGE_DIR", str(_DEFAULT_DB_DIR / "pdf_storage"))
)


def ensure_pdf_storage_dir() -> None:
    PDF_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# --- Wajib untuk Tahap 4 (sync Google Sheets & Drive) ---
GOOGLE_APPLICATION_CREDENTIALS = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
GOOGLE_SHEET_ID = os.environ.get("GOOGLE_SHEET_ID")
GOOGLE_DRIVE_FOLDER_ID = os.environ.get("GOOGLE_DRIVE_FOLDER_ID")


def validate_google_config() -> None:
    """Pastikan semua env var & file kredensial Google sudah tersedia
    sebelum mencoba konek ke Sheets/Drive. Lempar ConfigError dengan
    pesan jelas jika ada yang kurang, supaya gagal cepat & jelas
    (bukan error mentah dari Google API di tengah proses)."""
    missing = []
    if not GOOGLE_APPLICATION_CREDENTIALS:
        missing.append("GOOGLE_APPLICATION_CREDENTIALS")
    elif not Path(GOOGLE_APPLICATION_CREDENTIALS).is_file():
        raise ConfigError(
            f"GOOGLE_APPLICATION_CREDENTIALS diset ke '{GOOGLE_APPLICATION_CREDENTIALS}' "
            "tapi file tidak ditemukan di path tersebut."
        )
    if not GOOGLE_SHEET_ID:
        missing.append("GOOGLE_SHEET_ID")
    if not GOOGLE_DRIVE_FOLDER_ID:
        missing.append("GOOGLE_DRIVE_FOLDER_ID")
    if missing:
        raise ConfigError(
            "Environment variable berikut wajib diisi untuk sync Sheets/Drive: "
            + ", ".join(missing)
        )


def ensure_db_dir() -> None:
    """Pastikan folder tujuan file database sudah ada."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
