"""
Modul sync ke Google Sheets (Tahap 4).

ATURAN WAJIB (sesuai spesifikasi proyek):
  - Idempoten: cocokkan berdasarkan kolom "ID Nomor Surat" (document_id
    lokal). Fetch semua ID yang sudah ada di Sheet dulu, skip yang sudah
    ada.
  - APPEND-ONLY: tidak pernah menimpa/mengubah baris yang sudah ada.
  - Batch: pakai `append_rows` sekali panggil, BUKAN `append_row` satu-satu
    dalam loop.
"""
import logging

import gspread

from app.config import GOOGLE_SHEET_ID

logger = logging.getLogger(__name__)

# Urutan & nama kolom PERSIS sesuai skema Google Sheets tujuan yang
# ditentukan di spesifikasi proyek (27 kolom).
SHEET_HEADER = [
    "ID Nomor Surat", "Nomor Surat", "Jenis Kegiatan", "Provinsi Lokasi Kegiatan",
    "Pelaksanaan", "Tanggal Surat", "Jabatan Penanda Tangan", "Penanda Tangan",
    "Petugas 1 - Nama", "Petugas 1 - NIP", "Petugas 1 - No. Reg",
    "Petugas 2 - Nama", "Petugas 2 - NIP", "Petugas 2 - No. Reg",
    "Petugas 3 - Nama", "Petugas 3 - NIP", "Petugas 3 - No. Reg",
    "Petugas 4 - Nama", "Petugas 4 - NIP", "Petugas 4 - No. Reg",
    "Nama UPI", "Alamat UPI", "Jenis Produk / Grade",
    "Tanggal Diinput", "Nama File", "No. Reg", "Drive File URL",
]

# Kolom `extracted_fields` dalam urutan yang sama dengan SHEET_HEADER
# (setelah document_id di posisi pertama).
_ROW_FIELD_ORDER = [
    "nomor_surat", "jenis_kegiatan", "provinsi_lokasi", "pelaksanaan",
    "tanggal_surat", "jabatan_penandatangan", "penandatangan",
    "petugas1_nama", "petugas1_nip", "petugas1_no_reg",
    "petugas2_nama", "petugas2_nip", "petugas2_no_reg",
    "petugas3_nama", "petugas3_nip", "petugas3_no_reg",
    "petugas4_nama", "petugas4_nip", "petugas4_no_reg",
    "nama_upi", "alamat_upi", "jenis_produk_grade",
    "tanggal_diinput", "nama_file", "no_reg", "drive_file_url",
]


def get_sheets_client(creds) -> gspread.Client:
    return gspread.authorize(creds)


def get_worksheet(gc: gspread.Client):
    return gc.open_by_key(GOOGLE_SHEET_ID).sheet1


def ensure_header(worksheet) -> None:
    """Tulis baris header jika sheet masih kosong. Tidak pernah menimpa
    header yang sudah ada (append-only)."""
    first_row = worksheet.row_values(1)
    if not first_row:
        worksheet.append_row(SHEET_HEADER, value_input_option="USER_ENTERED")
        logger.info("Header Google Sheets ditulis (sheet sebelumnya kosong).")


def get_existing_ids(worksheet) -> set:
    """Ambil semua nilai kolom 'ID Nomor Surat' (kolom A) yang sudah ada,
    untuk pengecekan idempoten. Baris 1 (header) dilewati."""
    col_a = worksheet.col_values(1)
    return set(col_a[1:]) if len(col_a) > 1 else set()


def build_sheet_row(document_id: int, extracted_row) -> list:
    """Susun satu baris sesuai urutan SHEET_HEADER dari row extracted_fields
    (sqlite3.Row) + document_id sebagai kolom pertama."""
    values = [str(document_id)]
    for col in _ROW_FIELD_ORDER:
        value = extracted_row[col]
        values.append("" if value is None else str(value))
    return values


def append_rows_idempotent(worksheet, rows: list) -> tuple[int, int]:
    """Batch-append baris yang ID-nya BELUM ada di sheet. `rows` adalah
    list hasil build_sheet_row (kolom pertama = ID Nomor Surat sebagai
    string). Tidak pernah UPDATE/overwrite baris lama (append-only).

    Return (jumlah_baru_ditambahkan, jumlah_dilewati_karena_sudah_ada).
    """
    if not rows:
        return 0, 0

    existing_ids = get_existing_ids(worksheet)
    new_rows = [row for row in rows if row[0] not in existing_ids]
    skipped = len(rows) - len(new_rows)

    if new_rows:
        # SATU panggilan batch, bukan append_row per baris dalam loop.
        worksheet.append_rows(new_rows, value_input_option="USER_ENTERED")
        logger.info("Berhasil append %d baris baru ke Google Sheets.", len(new_rows))
    if skipped:
        logger.info(
            "%d baris dilewati (ID Nomor Surat sudah ada di Sheets, idempoten).", skipped
        )

    return len(new_rows), skipped
