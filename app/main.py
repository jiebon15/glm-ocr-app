"""
CLI Tahap 1: batch OCR dokumen PDF -> simpan ke SQLite lokal.

Contoh pemakaian:
    python -m app.main dokumen1.pdf dokumen2.pdf
    python -m app.main --input-dir ./surat_masuk/
    python -m app.main --input-dir ./surat_masuk/ --dpi 200

Alur per file:
  1. Hitung SHA-256 file -> jika sudah ada di DB (duplikat), skip.
  2. Insert baris baru ke tabel `documents` (status='pending').
  3. Konversi PDF -> gambar per halaman (pdf2image, butuh poppler-utils).
  4. OCR tiap gambar via model GLM-OCR (Ollama) -> simpan ke `ocr_results`.
  5. Update status dokumen: 'ocr_done' jika semua halaman sukses,
     'error' + pesan error jika gagal di tengah jalan.
"""
import argparse
import hashlib
import logging
import sys
from pathlib import Path
from typing import List

from app.db import (
    delete_ocr_results,
    find_document_by_hash,
    get_connection,
    init_db,
    insert_document,
    insert_ocr_result,
    reset_document_for_retry,
    update_document_status,
)
from app.ocr import get_ollama_client, ocr_image, pdf_to_images

# Status yang menandakan dokumen SUDAH pernah berhasil diproses sampai
# minimal tahap OCR. Hanya status inilah yang membuat file dianggap
# "duplikat" dan dilewati. Status lain (pending/error) berarti percobaan
# sebelumnya belum tuntas, sehingga akan di-retry otomatis.
_COMPLETED_STATUSES = {
    "ocr_done", "extracted", "reviewed", "uploaded_drive", "synced",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def collect_pdf_paths(files: List[str], input_dir: str | None) -> List[Path]:
    paths: List[Path] = []
    if input_dir:
        paths.extend(sorted(Path(input_dir).glob("*.pdf")))

    for f in files:
        p = Path(f)
        if p.is_dir():
            # Argumen positional ternyata folder (bukan --input-dir) ->
            # tetap expand isinya, bukan diperlakukan sebagai file tunggal.
            logger.info("Argumen '%s' adalah folder, membaca semua .pdf di dalamnya.", f)
            paths.extend(sorted(p.glob("*.pdf")))
        else:
            paths.append(p)

    # Dedup path yang sama sambil mempertahankan urutan
    seen = set()
    unique_paths = []
    for p in paths:
        resolved = p.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_paths.append(p)
    return unique_paths


def process_pdf(pdf_path: Path, dpi: int) -> str:
    """Proses satu file PDF. Mengembalikan status akhir: ocr_done/skip/error."""
    if not pdf_path.is_file():
        logger.error("File tidak ditemukan: %s", pdf_path)
        return "error"

    file_hash = compute_sha256(pdf_path)

    with get_connection() as conn:
        existing = find_document_by_hash(conn, file_hash)
        if existing is not None and existing["status"] in _COMPLETED_STATUSES:
            logger.info(
                "Lewati (duplikat, sudah pernah selesai diproses sbg id=%s, status=%s): %s",
                existing["id"],
                existing["status"],
                pdf_path.name,
            )
            return "skip"
        elif existing is not None:
            # Percobaan sebelumnya belum tuntas (pending/error) -> retry
            # memakai document_id yang sama, bersihkan hasil OCR lama.
            document_id = existing["id"]
            logger.info(
                "Mengulangi proses (percobaan sebelumnya status=%s): %s",
                existing["status"],
                pdf_path.name,
            )
            delete_ocr_results(conn, document_id)
            reset_document_for_retry(conn, document_id)
        else:
            document_id = insert_document(conn, pdf_path.name, file_hash)

    logger.info("Mulai proses: %s (document_id=%s)", pdf_path.name, document_id)

    client = get_ollama_client()
    try:
        page_count = 0
        for page_number, image in enumerate(pdf_to_images(pdf_path, dpi=dpi), start=1):
            logger.info("  OCR halaman %d ...", page_number)
            text = ocr_image(image, client)
            with get_connection() as conn:
                insert_ocr_result(conn, document_id, page_number, text)
            page_count += 1

        if page_count == 0:
            raise RuntimeError("PDF tidak menghasilkan halaman apa pun (kosong/rusak?)")

        with get_connection() as conn:
            update_document_status(conn, document_id, "ocr_done")
        logger.info("Selesai: %s (%d halaman)", pdf_path.name, page_count)
        return "ocr_done"

    except Exception as exc:  # noqa: BLE001 - ingin menangkap semua error runtime OCR
        logger.exception("Gagal memproses %s", pdf_path.name)
        with get_connection() as conn:
            update_document_status(conn, document_id, "error", error_message=str(exc))
        return "error"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tahap 1: Batch OCR PDF -> SQLite (GLM-OCR via Ollama)."
    )
    parser.add_argument("files", nargs="*", help="Path file PDF individual")
    parser.add_argument(
        "--input-dir", help="Folder berisi file .pdf untuk diproses secara batch"
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="Resolusi konversi PDF->gambar (default: 300)"
    )
    args = parser.parse_args()

    pdf_paths = collect_pdf_paths(args.files, args.input_dir)
    if not pdf_paths:
        parser.error("Tidak ada file PDF yang diberikan. Gunakan argumen file atau --input-dir.")

    init_db()

    results = {"ocr_done": 0, "skip": 0, "error": 0}
    for pdf_path in pdf_paths:
        status = process_pdf(pdf_path, dpi=args.dpi)
        results[status] = results.get(status, 0) + 1

    logger.info(
        "=== Ringkasan: %d berhasil, %d dilewati (duplikat), %d gagal dari %d file ===",
        results["ocr_done"],
        results["skip"],
        results["error"],
        len(pdf_paths),
    )
    return 0 if results["error"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
