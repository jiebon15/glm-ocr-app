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


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, coltype: str) -> None:
    """Tambahkan kolom ke tabel jika belum ada (migrasi ringan, aman
    dipanggil berulang kali, tidak menyentuh data yang sudah ada)."""
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")


def init_db() -> None:
    """Buat semua tabel jika belum ada, lalu jalankan migrasi ringan.
    Aman dipanggil berulang kali."""
    with get_connection() as conn:
        conn.executescript(SCHEMA)
        # Tahap 3 (GUI): perlu tahu lokasi file PDF asli di disk (untuk
        # retry & upload Drive di Tahap 4). Migrasi non-destruktif -
        # database lama tanpa kolom ini tetap aman, nilainya NULL.
        _ensure_column(conn, "documents", "local_path", "TEXT")


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


def insert_document(
    conn: sqlite3.Connection,
    file_name: str,
    file_hash: str,
    local_path: Optional[str] = None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO documents (file_name, file_hash, upload_time, status, local_path)
        VALUES (?, ?, ?, 'pending', ?)
        """,
        (file_name, file_hash, datetime.now().isoformat(), local_path),
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


def get_status_counts(conn: sqlite3.Connection) -> dict:
    """Hitung jumlah dokumen per status, untuk ringkasan di GUI."""
    rows = conn.execute("SELECT status, COUNT(*) AS n FROM documents GROUP BY status").fetchall()
    return {row["status"]: row["n"] for row in rows}


def get_all_documents_overview(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Daftar semua dokumen (untuk tab Ringkasan/Riwayat di GUI)."""
    return conn.execute(
        "SELECT id, file_name, status, error_message, upload_time FROM documents ORDER BY id DESC"
    ).fetchall()


def has_ocr_results(conn: sqlite3.Connection, document_id: int) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM ocr_results WHERE document_id = ? LIMIT 1", (document_id,)
    )
    return cur.fetchone() is not None


def get_review_queue(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Dokumen berstatus 'extracted' beserta field hasil ekstraksinya,
    siap ditampilkan di tabel editable untuk verifikasi manual."""
    columns = ", ".join(f"ef.{c}" for c in _EXTRACTED_FIELD_COLUMNS)
    return conn.execute(
        f"""
        SELECT d.id AS document_id, d.file_name, {columns}
        FROM documents d
        JOIN extracted_fields ef ON ef.document_id = d.id
        WHERE d.status = 'extracted'
        ORDER BY d.id
        """
    ).fetchall()


def update_extracted_fields(conn: sqlite3.Connection, document_id: int, fields: dict) -> None:
    """Terapkan koreksi manual dari GUI ke tabel extracted_fields.
    `fields` boleh berisi subset _EXTRACTED_FIELD_COLUMNS; hanya key yang
    ada yang di-update."""
    cols_to_update = [c for c in _EXTRACTED_FIELD_COLUMNS if c in fields]
    if not cols_to_update:
        return
    set_clause = ", ".join(f"{c} = ?" for c in cols_to_update)
    values = [fields[c] for c in cols_to_update] + [document_id]
    conn.execute(
        f"UPDATE extracted_fields SET {set_clause} WHERE document_id = ?", values
    )


def mark_reviewed(conn: sqlite3.Connection, document_id: int) -> None:
    """Tandai dokumen sudah diverifikasi manual (Tahap 3). Sinkronisasi
    ke Sheets/Drive baru terjadi di Tahap 4."""
    update_document_status(conn, document_id, "reviewed")


def get_extracted_fields_by_document(conn: sqlite3.Connection, document_id: int) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM extracted_fields WHERE document_id = ?", (document_id,)
    ).fetchone()


def update_drive_file_url(conn: sqlite3.Connection, document_id: int, url: str) -> None:
    conn.execute(
        "UPDATE extracted_fields SET drive_file_url = ? WHERE document_id = ?",
        (url, document_id),
    )


def clear_local_path(conn: sqlite3.Connection, document_id: int) -> None:
    """Kosongkan local_path setelah file PDF asli dihapus (sumber
    kebenaran berpindah ke Drive, tidak disimpan ganda secara lokal)."""
    conn.execute("UPDATE documents SET local_path = NULL WHERE id = ?", (document_id,))


def mark_synced(conn: sqlite3.Connection, document_id: int) -> None:
    """Tandai dokumen sudah tersinkron ke Google Sheets."""
    conn.execute(
        "UPDATE extracted_fields SET synced = 1, synced_at = ? WHERE document_id = ?",
        (datetime.now().isoformat(), document_id),
    )
    update_document_status(conn, document_id, "synced")


def get_unsynced_uploaded_documents(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Dokumen yang sudah ter-upload ke Drive tapi belum tersinkron ke
    Sheets (termasuk sisa dari percobaan sync sebelumnya yang gagal
    di tengah jalan)."""
    return conn.execute(
        """
        SELECT d.id AS document_id, d.file_name, d.status
        FROM documents d
        JOIN extracted_fields ef ON ef.document_id = d.id
        WHERE d.status = 'uploaded_drive' AND (ef.synced IS NULL OR ef.synced = 0)
        ORDER BY d.id
        """
    ).fetchall()
