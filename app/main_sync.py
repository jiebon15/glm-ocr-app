"""
CLI Tahap 4: upload PDF ke Google Drive + sync field ke Google Sheets,
untuk dokumen yang sudah diverifikasi manual (status 'reviewed').

Trigger manual (belum auto-sync, itu Tahap 5):
    python -m app.main_sync

Alur:
  1. Cek konfigurasi Google (credentials, Sheet ID, Drive folder ID) ada.
  2. Cek koneksi internet. Kalau offline, berhenti (status dokumen tetap
     'reviewed', bisa dicoba lagi nanti tanpa efek samping).
  3. Untuk tiap dokumen 'reviewed': upload PDF asli ke Drive (folder flat),
     simpan URL ke extracted_fields.drive_file_url, update status jadi
     'uploaded_drive', lalu HAPUS file PDF lokal (sumber kebenaran pindah
     ke Drive).
  4. Kumpulkan SEMUA dokumen berstatus 'uploaded_drive' yang belum synced
     (termasuk sisa gagal dari run sebelumnya) -> sync ke Sheets sekaligus
     dalam SATU batch append_rows, idempoten berdasarkan ID Nomor Surat,
     APPEND-ONLY (tidak pernah menimpa baris lama).
  5. Tandai status 'synced' untuk dokumen yang berhasil (baik baru
     ditambahkan maupun ternyata sudah ada di Sheets sebelumnya).
"""
import logging
import sys
from pathlib import Path

from app.config import ConfigError, validate_google_config
from app.db import (
    clear_local_path,
    get_connection,
    get_document_by_id,
    get_documents_by_status,
    get_extracted_fields_by_document,
    get_unsynced_uploaded_documents,
    init_db,
    mark_synced,
    update_document_status,
    update_drive_file_url,
)
from app.gdrive import get_drive_service, get_google_credentials, is_online, upload_pdf_to_drive
from app.gsheets import append_rows_idempotent, build_sheet_row, ensure_header, get_sheets_client, get_worksheet

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def upload_reviewed_documents_to_drive(drive_service) -> dict:
    """Tahap upload Drive untuk semua dokumen 'reviewed'. Return ringkasan
    jumlah per hasil."""
    with get_connection() as conn:
        targets = get_documents_by_status(conn, "reviewed")

    results = {"uploaded_drive": 0, "error": 0}
    for doc in targets:
        document_id = doc["id"]
        local_path = doc["local_path"]

        if not local_path or not Path(local_path).is_file():
            logger.error(
                "File PDF lokal tidak ditemukan untuk document_id=%s (%s): %s",
                document_id, doc["file_name"], local_path,
            )
            with get_connection() as conn:
                update_document_status(
                    conn, document_id, "error",
                    error_message=f"File PDF lokal tidak ditemukan: {local_path}",
                )
            results["error"] += 1
            continue

        try:
            logger.info("Upload ke Drive: %s (document_id=%s)", doc["file_name"], document_id)
            url = upload_pdf_to_drive(drive_service, Path(local_path), doc["file_name"])
            with get_connection() as conn:
                update_drive_file_url(conn, document_id, url)
                update_document_status(conn, document_id, "uploaded_drive")
            Path(local_path).unlink(missing_ok=True)
            with get_connection() as conn:
                clear_local_path(conn, document_id)
            logger.info("Selesai upload Drive & hapus file lokal: %s", doc["file_name"])
            results["uploaded_drive"] += 1
        except Exception as exc:  # noqa: BLE001
            logger.exception("Gagal upload Drive: %s", doc["file_name"])
            with get_connection() as conn:
                update_document_status(conn, document_id, "error", error_message=str(exc))
            results["error"] += 1

    return results


def sync_to_sheets(sheets_worksheet) -> dict:
    """Kumpulkan semua dokumen uploaded_drive yang belum synced, sync
    sekaligus dalam satu batch (idempoten, append-only)."""
    with get_connection() as conn:
        targets = get_unsynced_uploaded_documents(conn)
        rows = []
        for doc in targets:
            fields = get_extracted_fields_by_document(conn, doc["document_id"])
            if fields is None:
                logger.error(
                    "Tidak ada extracted_fields untuk document_id=%s, dilewati dari sync.",
                    doc["document_id"],
                )
                continue
            rows.append(build_sheet_row(doc["document_id"], fields))

    if not rows:
        logger.info("Tidak ada dokumen yang perlu disinkronkan ke Sheets.")
        return {"synced": 0}

    n_new, n_skipped = append_rows_idempotent(sheets_worksheet, rows)

    # Baik baris yang baru ditambahkan maupun yang ternyata sudah ada di
    # Sheets (idempoten) -> keduanya berarti data SUDAH ada di Sheets,
    # jadi tandai synced di DB lokal untuk kedua kasus.
    with get_connection() as conn:
        for doc in targets:
            mark_synced(conn, doc["document_id"])

    logger.info(
        "Sync Sheets selesai: %d baris baru, %d sudah ada sebelumnya (idempoten). Total %d dokumen ditandai synced.",
        n_new, n_skipped, len(targets),
    )
    return {"synced": len(targets), "new_rows": n_new, "already_existed": n_skipped}


def main() -> int:
    init_db()

    try:
        validate_google_config()
    except ConfigError as exc:
        logger.error("Konfigurasi Google belum lengkap: %s", exc)
        return 1

    if not is_online():
        logger.error(
            "Tidak ada koneksi internet (gagal konek ke sheets.googleapis.com). "
            "Dokumen tetap berstatus 'reviewed', coba lagi nanti."
        )
        return 1

    creds = get_google_credentials()
    drive_service = get_drive_service(creds)
    sheets_client = get_sheets_client(creds)
    worksheet = get_worksheet(sheets_client)
    ensure_header(worksheet)

    drive_results = upload_reviewed_documents_to_drive(drive_service)
    sync_results = sync_to_sheets(worksheet)

    logger.info(
        "=== Ringkasan: %d upload Drive sukses, %d error upload, %d dokumen tersinkron ke Sheets "
        "(%s baris baru, %s sudah ada sebelumnya) ===",
        drive_results["uploaded_drive"], drive_results["error"],
        sync_results.get("synced", 0),
        sync_results.get("new_rows", "-"), sync_results.get("already_existed", "-"),
    )
    return 0 if drive_results["error"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
