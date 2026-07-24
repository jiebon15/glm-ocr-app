"""
CLI Tahap 2: ekstraksi field terstruktur dari dokumen berstatus 'ocr_done'.

Contoh pemakaian:
    python -m app.main_extract                 # proses semua dokumen ocr_done
    python -m app.main_extract --document-id 3  # proses satu dokumen spesifik
                                                  # (retry meski statusnya bukan ocr_done)

Alur per dokumen (lihat app/extraction.py untuk arsitektur dua-panggilan):
  1. Ambil teks OCR halaman 1-2 (untuk field administratif) DAN teks OCR
     seluruh halaman (untuk field lain: petugas/UPI/no_reg, termasuk
     lampiran).
  2. Kirim keduanya ke model ekstraksi (qwen2.5:7b via Ollama, dua
     panggilan constrained JSON terpisah) -> hasil digabung jadi 23 field.
  3. Simpan hasil ke tabel `extracted_fields`.
  4. Update status dokumen -> 'extracted', atau 'error' + pesan jika gagal.
"""
import argparse
import logging
import sys

from app.config import EXTRACTION_MAX_PAGES
from app.db import (
    delete_extracted_fields,
    get_combined_ocr_text,
    get_connection,
    get_document_by_id,
    get_documents_by_status,
    init_db,
    insert_extracted_fields,
    update_document_status,
)
from app.extraction import extract_fields, get_extraction_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def process_document(document_id: int, file_name: str) -> str:
    """Proses ekstraksi satu dokumen. Mengembalikan status akhir."""
    with get_connection() as conn:
        admin_text = get_combined_ocr_text(conn, document_id, max_pages=EXTRACTION_MAX_PAGES)
        full_text = get_combined_ocr_text(conn, document_id, max_pages=None)

    if not full_text.strip():
        logger.error("Tidak ada teks OCR untuk document_id=%s, dilewati.", document_id)
        with get_connection() as conn:
            update_document_status(
                conn, document_id, "error", error_message="Teks OCR kosong, tidak bisa diekstrak."
            )
        return "error"

    logger.info(
        "Ekstraksi field: %s (document_id=%s, halaman admin<=%s, halaman full=semua)",
        file_name, document_id, EXTRACTION_MAX_PAGES or "semua",
    )
    client = get_extraction_client()
    try:
        fields = extract_fields(admin_text, full_text, client, file_name=file_name)
        with get_connection() as conn:
            # Bersihkan hasil ekstraksi lama jika ini retry
            delete_extracted_fields(conn, document_id)
            insert_extracted_fields(conn, document_id, fields, nama_file=file_name)
            update_document_status(conn, document_id, "extracted")
        logger.info("Selesai ekstraksi: %s", file_name)
        return "extracted"
    except Exception as exc:  # noqa: BLE001
        logger.exception("Gagal mengekstrak field: %s", file_name)
        with get_connection() as conn:
            update_document_status(conn, document_id, "error", error_message=str(exc))
        return "error"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tahap 2: Ekstraksi field terstruktur dari hasil OCR."
    )
    parser.add_argument(
        "--document-id",
        type=int,
        default=None,
        help="Proses satu document_id spesifik (retry), abaikan filter status.",
    )
    args = parser.parse_args()

    init_db()

    with get_connection() as conn:
        if args.document_id is not None:
            doc = get_document_by_id(conn, args.document_id)
            if doc is None:
                logger.error("document_id=%s tidak ditemukan di database.", args.document_id)
                return 1
            targets = [doc]
        else:
            targets = get_documents_by_status(conn, "ocr_done")

    if not targets:
        logger.info("Tidak ada dokumen berstatus 'ocr_done' untuk diekstrak.")
        return 0

    results = {"extracted": 0, "error": 0}
    for doc in targets:
        status = process_document(doc["id"], doc["file_name"])
        results[status] = results.get(status, 0) + 1

    logger.info(
        "=== Ringkasan: %d berhasil, %d gagal dari %d dokumen ===",
        results["extracted"],
        results["error"],
        len(targets),
    )
    return 0 if results["error"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
