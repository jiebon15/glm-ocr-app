"""
Layer database SQLite.

Membuat skema penuh (documents, ocr_results, extracted_fields) sesuai
rancangan proyek, meskipun Tahap 1 hanya memakai `documents` dan
`ocr_results`. Tabel `extracted_fields` disiapkan lebih dulu supaya Tahap 2
tidak perlu migrasi skema.
"""
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from app.config import DB_PATH, ensure_db_dir

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name TEXT NOT NULL,
    file_hash TEXT NOT NULL UNIQUE,
    upload_time DATETIME NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending','ocr_done','extracted','reviewed',
                           'uploaded_drive','synced','error')),
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS ocr_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    raw_text TEXT,
    ocr_time DATETIME NOT NULL
);

CREATE TABLE IF NOT EXISTS extracted_fields (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    nomor_surat TEXT,
    jenis_kegiatan TEXT,
    provinsi_lokasi TEXT,
    pelaksanaan TEXT,
    tanggal_surat DATE,
    jabatan_penandatangan TEXT,
    penandatangan TEXT,
    petugas1_nama TEXT,
    petugas1_nip TEXT,
    petugas1_no_reg TEXT,
    petugas2_nama TEXT,
    petugas2_nip TEXT,
    petugas2_no_reg TEXT,
    petugas3_nama TEXT,
    petugas3_nip TEXT,
    petugas3_no_reg TEXT,
    petugas4_nama TEXT,
    petugas4_nip TEXT,
    petugas4_no_reg TEXT,
    nama_upi TEXT,
    alamat_upi TEXT,
    jenis_produk_grade TEXT,
    tanggal_diinput DATETIME DEFAULT CURRENT_TIMESTAMP,
    nama_file TEXT,
    no_reg TEXT,
    drive_file_url TEXT,
    synced BOOLEAN NOT NULL DEFAULT 0,
    synced_at DATETIME
);

CREATE INDEX IF NOT EXISTS idx_ocr_results_document_id
    ON ocr_results(document_id);
CREATE INDEX IF NOT EXISTS idx_extracted_fields_document_id
    ON extracted_fields(document_id);
"""


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """Context manager koneksi SQLite dengan foreign_keys aktif."""
    ensure_db_dir()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Buat semua tabel jika belum ada. Aman dipanggil berulang kali."""
    with get_connection() as conn:
        conn.executescript(SCHEMA)


def find_document_by_hash(conn: sqlite3.Connection, file_hash: str) -> Optional[sqlite3.Row]:
    cur = conn.execute("SELECT * FROM documents WHERE file_hash = ?", (file_hash,))
    return cur.fetchone()


def delete_ocr_results(conn: sqlite3.Connection, document_id: int) -> None:
    """Hapus semua hasil OCR lama milik satu dokumen (dipakai sebelum retry)."""
    conn.execute("DELETE FROM ocr_results WHERE document_id = ?", (document_id,))


def reset_document_for_retry(conn: sqlite3.Connection, document_id: int) -> None:
    """Reset status dokumen ke 'pending' sebelum diproses ulang."""
    conn.execute(
        "UPDATE documents SET status = 'pending', error_message = NULL WHERE id = ?",
        (document_id,),
    )


def insert_document(conn: sqlite3.Connection, file_name: str, file_hash: str) -> int:
    cur = conn.execute(
        """
        INSERT INTO documents (file_name, file_hash, upload_time, status)
        VALUES (?, ?, ?, 'pending')
        """,
        (file_name, file_hash, datetime.now().isoformat()),
    )
    return cur.lastrowid


def update_document_status(
    conn: sqlite3.Connection,
    document_id: int,
    status: str,
    error_message: Optional[str] = None,
) -> None:
    conn.execute(
        "UPDATE documents SET status = ?, error_message = ? WHERE id = ?",
        (status, error_message, document_id),
    )


def insert_ocr_result(
    conn: sqlite3.Connection,
    document_id: int,
    page_number: int,
    raw_text: str,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO ocr_results (document_id, page_number, raw_text, ocr_time)
        VALUES (?, ?, ?, ?)
        """,
        (document_id, page_number, raw_text, datetime.now().isoformat()),
    )
    return cur.lastrowid


def get_documents_by_status(conn: sqlite3.Connection, status: str) -> list[sqlite3.Row]:
    cur = conn.execute(
        "SELECT * FROM documents WHERE status = ? ORDER BY id", (status,)
    )
    return cur.fetchall()


def get_document_by_id(conn: sqlite3.Connection, document_id: int) -> Optional[sqlite3.Row]:
    cur = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,))
    return cur.fetchone()


def get_combined_ocr_text(
    conn: sqlite3.Connection,
    document_id: int,
    max_pages: Optional[int] = None,
) -> str:
    """Gabungkan raw_text halaman satu dokumen, urut per halaman, dengan
    penanda batas halaman agar model ekstraksi tetap punya konteks struktur
    multi-halaman.

    `max_pages`: jika diisi, hanya ambil N halaman pertama (mis. untuk
    mengecualikan lampiran di halaman-halaman belakang yang sering tidak
    diverifikasi ketat oleh pembuat/validator surat, sehingga rawan jadi
    sumber salah ekstraksi)."""
    cur = conn.execute(
        """
        SELECT page_number, raw_text FROM ocr_results
        WHERE document_id = ? ORDER BY page_number
        """,
        (document_id,),
    )
    pages = cur.fetchall()
    if max_pages is not None:
        pages = pages[:max_pages]
    parts = [f"--- Halaman {row['page_number']} ---\n{row['raw_text']}" for row in pages]
    return "\n\n".join(parts)


def delete_extracted_fields(conn: sqlite3.Connection, document_id: int) -> None:
    """Hapus hasil ekstraksi lama milik satu dokumen (dipakai sebelum retry)."""
    conn.execute("DELETE FROM extracted_fields WHERE document_id = ?", (document_id,))


# Kolom hasil ekstraksi LLM (tidak termasuk id/document_id/tanggal_diinput/
# nama_file/drive_file_url/synced*, yang diisi dari sumber lain — lihat
# app/extraction.py untuk skema JSON yang dikirim ke model).
_EXTRACTED_FIELD_COLUMNS = [
    "nomor_surat", "jenis_kegiatan", "provinsi_lokasi", "pelaksanaan",
    "tanggal_surat", "jabatan_penandatangan", "penandatangan",
    "petugas1_nama", "petugas1_nip", "petugas1_no_reg",
    "petugas2_nama", "petugas2_nip", "petugas2_no_reg",
    "petugas3_nama", "petugas3_nip", "petugas3_no_reg",
    "petugas4_nama", "petugas4_nip", "petugas4_no_reg",
    "nama_upi", "alamat_upi", "jenis_produk_grade", "no_reg",
]


def insert_extracted_fields(
    conn: sqlite3.Connection,
    document_id: int,
    fields: dict,
    nama_file: str,
) -> int:
    """Simpan hasil ekstraksi LLM ke tabel extracted_fields.
    `fields` harus berisi persis kolom di _EXTRACTED_FIELD_COLUMNS
    (nilai boleh None); nilai lain diabaikan."""
    columns = _EXTRACTED_FIELD_COLUMNS + ["nama_file", "document_id"]
    values = [fields.get(col) for col in _EXTRACTED_FIELD_COLUMNS] + [nama_file, document_id]
    placeholders = ", ".join(["?"] * len(columns))
    column_list = ", ".join(columns)
    cur = conn.execute(
        f"INSERT INTO extracted_fields ({column_list}) VALUES ({placeholders})",
        values,
    )
    return cur.lastrowid
