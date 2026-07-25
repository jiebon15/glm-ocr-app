"""
GUI Streamlit Tahap 3.

Menjalankan:
    streamlit run app/streamlit_app.py

Fitur:
  - Upload PDF batch, jalankan OCR (Tahap 1) + ekstraksi (Tahap 2) langsung
    dari GUI.
  - Tabel editable (st.data_editor) untuk verifikasi/koreksi manual field
    hasil ekstraksi, terutama nomor surat, NIP, No. Reg yang rawan salah.
  - Tombol "Tandai Reviewed" per baris (checkbox + terapkan) — BELUM
    melakukan sync ke Google Sheets/Drive, itu baru di Tahap 4.
  - Tab Ringkasan/Riwayat: status semua dokumen, retry untuk yang error.

CATATAN: OCR & ekstraksi dijalankan SINKRON di proses Streamlit (bukan
background job) karena aplikasi ini single-user, lokal, dan modelnya
berjalan di GPU yang sama — menjalankan job paralel/background berisiko
rebutan VRAM. Untuk batch besar, proses akan memblokir UI selama
berjalan (ditandai spinner), ini disengaja demi kesederhanaan.
"""
import hashlib
import logging
import sys
from pathlib import Path

# Streamlit menjalankan file ini langsung (bukan lewat `python -m`), jadi
# secara default Python hanya tahu folder app/ ini, bukan folder root
# proyek tempat package `app` (dan modul lain di app/) diimpor. Tambahkan
# manual supaya `streamlit run app/streamlit_app.py` selalu jalan, dari
# direktori kerja mana pun.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st

from app.config import ConfigError, PDF_STORAGE_DIR, ensure_pdf_storage_dir, validate_google_config
from app.db import (
    get_all_documents_overview,
    get_combined_ocr_text,
    get_connection,
    get_document_by_id,
    get_review_queue,
    get_status_counts,
    has_ocr_results,
    init_db,
    mark_reviewed,
    update_extracted_fields,
)
from app.main import compute_sha256, process_pdf
from app.main_extract import process_document
from app.main_sync import sync_to_sheets, upload_reviewed_documents_to_drive
from app.gdrive import get_drive_service, get_google_credentials, is_online
from app.gsheets import ensure_header, get_sheets_client, get_worksheet

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(page_title="OCR Dokumen Dinas", layout="wide")

init_db()
ensure_pdf_storage_dir()

# Kolom yang ditampilkan & bisa diedit di tabel review, dengan label
# ramah-pengguna. Urutan sengaja menaruh field kritikal (nomor surat, NIP,
# No. Reg) lebih dulu agar mudah dicek.
_REVIEW_COLUMN_LABELS = {
    "nomor_surat": "Nomor Surat",
    "no_reg": "No. Reg (Dokumen)",
    "tanggal_surat": "Tanggal Surat",
    "jenis_kegiatan": "Jenis Kegiatan",
    "provinsi_lokasi": "Provinsi Lokasi",
    "pelaksanaan": "Pelaksanaan",
    "jabatan_penandatangan": "Jabatan Penandatangan",
    "penandatangan": "Penandatangan",
    "petugas1_nama": "Petugas 1 - Nama",
    "petugas1_nip": "Petugas 1 - NIP",
    "petugas1_no_reg": "Petugas 1 - No. Reg",
    "petugas2_nama": "Petugas 2 - Nama",
    "petugas2_nip": "Petugas 2 - NIP",
    "petugas2_no_reg": "Petugas 2 - No. Reg",
    "petugas3_nama": "Petugas 3 - Nama",
    "petugas3_nip": "Petugas 3 - NIP",
    "petugas3_no_reg": "Petugas 3 - No. Reg",
    "petugas4_nama": "Petugas 4 - Nama",
    "petugas4_nip": "Petugas 4 - NIP",
    "petugas4_no_reg": "Petugas 4 - No. Reg",
    "nama_upi": "Nama UPI",
    "alamat_upi": "Alamat UPI",
    "jenis_produk_grade": "Jenis Produk / Grade",
}
_EDITABLE_COLS = list(_REVIEW_COLUMN_LABELS.keys())


def _save_uploaded_file(uploaded_file) -> Path:
    """Simpan file upload Streamlit ke PDF_STORAGE_DIR, hindari tabrakan
    nama dengan menambah suffix hash pendek jika perlu."""
    dest = PDF_STORAGE_DIR / uploaded_file.name
    content = uploaded_file.getvalue()
    if dest.exists():
        existing_hash = compute_sha256(dest)
        new_hash = hashlib.sha256(content).hexdigest()
        if existing_hash != new_hash:
            short = new_hash[:8]
            dest = PDF_STORAGE_DIR / f"{dest.stem}_{short}{dest.suffix}"
    dest.write_bytes(content)
    return dest


def _run_pipeline_for_file(uploaded_file, dpi: int) -> dict:
    """OCR + ekstraksi untuk satu file upload. Return dict ringkasan."""
    saved_path = _save_uploaded_file(uploaded_file)
    ocr_status = process_pdf(saved_path, dpi=dpi)

    if ocr_status == "error":
        return {"file": uploaded_file.name, "ocr": "error", "extract": "-"}
    if ocr_status == "skip":
        return {"file": uploaded_file.name, "ocr": "skip (duplikat)", "extract": "-"}

    # ocr_status == 'ocr_done' -> lanjut ekstraksi
    file_hash = compute_sha256(saved_path)
    with get_connection() as conn:
        from app.db import find_document_by_hash
        doc = find_document_by_hash(conn, file_hash)
    extract_status = process_document(doc["id"], doc["file_name"])
    return {"file": uploaded_file.name, "ocr": ocr_status, "extract": extract_status}


def render_upload_tab():
    st.subheader("Upload & Proses Batch")
    st.caption(
        "Upload satu atau beberapa PDF. Tiap file akan di-OCR (GLM-OCR) lalu "
        "diekstrak field-nya (qwen2.5) secara berurutan."
    )

    uploaded_files = st.file_uploader(
        "Pilih file PDF", type=["pdf"], accept_multiple_files=True
    )
    dpi = st.number_input(
        "DPI konversi PDF -> gambar",
        min_value=100, max_value=400, value=150, step=25,
        help="Turunkan jika mengalami CUDA out of memory saat OCR.",
    )

    if st.button("Mulai OCR + Ekstraksi", type="primary", disabled=not uploaded_files):
        results = []
        progress = st.progress(0.0, text="Memulai...")
        for i, uploaded_file in enumerate(uploaded_files):
            progress.progress(
                i / len(uploaded_files), text=f"Memproses {uploaded_file.name} ..."
            )
            try:
                results.append(_run_pipeline_for_file(uploaded_file, dpi))
            except Exception as exc:  # noqa: BLE001
                logger.exception("Gagal memproses %s", uploaded_file.name)
                results.append({"file": uploaded_file.name, "ocr": "error", "extract": str(exc)})
        progress.progress(1.0, text="Selesai.")

        st.write("### Hasil")
        st.dataframe(pd.DataFrame(results), width='stretch', hide_index=True)

        n_ok = sum(1 for r in results if r["extract"] == "extracted")
        n_skip = sum(1 for r in results if r["ocr"] == "skip (duplikat)")
        n_err = len(results) - n_ok - n_skip
        st.success(f"{n_ok} berhasil diekstrak, {n_skip} dilewati (duplikat), {n_err} gagal.")
        if n_ok > 0:
            st.info("Buka tab **Verifikasi & Edit** untuk mengecek hasilnya.")


def render_review_tab():
    st.subheader("Verifikasi & Edit Manual")
    st.caption(
        "Periksa terutama Nomor Surat, NIP, dan No. Reg — field ini rawan "
        "salah OCR/ekstraksi. Edit langsung di tabel, lalu simpan."
    )

    with get_connection() as conn:
        queue = get_review_queue(conn)

    if not queue:
        st.info("Tidak ada dokumen menunggu review saat ini.")
        return

    df = pd.DataFrame([dict(row) for row in queue])
    df["Sudah Direview?"] = False

    display_df = df[["document_id", "file_name"] + _EDITABLE_COLS + ["Sudah Direview?"]].rename(
        columns={"document_id": "ID", "file_name": "Nama File", **_REVIEW_COLUMN_LABELS}
    )

    edited_df = st.data_editor(
        display_df,
        width='stretch',
        hide_index=True,
        disabled=["ID", "Nama File"],
        key="review_editor",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Simpan Perubahan Field", type="primary"):
            _reverse_labels = {v: k for k, v in _REVIEW_COLUMN_LABELS.items()}
            n_saved = 0
            with get_connection() as conn:
                for _, row in edited_df.iterrows():
                    document_id = int(row["ID"])
                    fields = {
                        _reverse_labels[label]: (row[label] if pd.notna(row[label]) else None)
                        for label in _REVIEW_COLUMN_LABELS.values()
                    }
                    update_extracted_fields(conn, document_id, fields)
                    n_saved += 1
            st.success(f"Perubahan tersimpan untuk {n_saved} dokumen.")
            st.rerun()

    with col2:
        if st.button("✅ Tandai 'Sudah Direview' (baris tercentang)"):
            to_mark = edited_df[edited_df["Sudah Direview?"] == True]  # noqa: E712
            if to_mark.empty:
                st.warning("Tidak ada baris yang dicentang 'Sudah Direview?'.")
            else:
                with get_connection() as conn:
                    for _, row in to_mark.iterrows():
                        mark_reviewed(conn, int(row["ID"]))
                st.success(f"{len(to_mark)} dokumen ditandai reviewed.")
                st.rerun()

    st.divider()
    st.caption("Cek teks OCR mentah untuk verifikasi silang (opsional):")
    doc_options = {f"{row['file_name']} (id={row['document_id']})": row["document_id"] for row in queue}
    selected_label = st.selectbox("Pilih dokumen", options=list(doc_options.keys()))
    if selected_label:
        with get_connection() as conn:
            ocr_text = get_combined_ocr_text(conn, doc_options[selected_label])
        with st.expander("Lihat teks OCR mentah (seluruh halaman)"):
            st.text(ocr_text)


def render_summary_tab():
    st.subheader("Ringkasan & Riwayat Dokumen")

    with get_connection() as conn:
        counts = get_status_counts(conn)
        overview = get_all_documents_overview(conn)

    cols = st.columns(6)
    status_order = ["pending", "ocr_done", "extracted", "reviewed", "uploaded_drive", "synced"]
    for col, status in zip(cols, status_order):
        col.metric(status, counts.get(status, 0))
    if counts.get("error"):
        st.error(f"⚠️ {counts['error']} dokumen berstatus ERROR — lihat tabel di bawah.")

    st.divider()

    overview_df = pd.DataFrame([dict(row) for row in overview])
    st.dataframe(overview_df, width='stretch', hide_index=True)

    st.divider()
    st.write("#### Retry dokumen bermasalah")
    error_docs = [row for row in overview if row["status"] == "error"]
    if not error_docs:
        st.caption("Tidak ada dokumen berstatus error.")
        return

    doc_labels = {f"{row['file_name']} (id={row['id']})": row["id"] for row in error_docs}
    selected = st.selectbox("Pilih dokumen untuk di-retry", options=list(doc_labels.keys()))
    if st.button("🔁 Retry"):
        document_id = doc_labels[selected]
        with get_connection() as conn:
            doc = get_document_by_id(conn, document_id)
            already_ocr = has_ocr_results(conn, document_id)

        with st.spinner("Memproses ulang..."):
            if already_ocr:
                # OCR sebelumnya sukses, yang gagal di tahap ekstraksi
                result = process_document(document_id, doc["file_name"])
                st.write(f"Hasil retry ekstraksi: **{result}**")
            elif doc["local_path"] and Path(doc["local_path"]).exists():
                result = process_pdf(Path(doc["local_path"]), dpi=150)
                st.write(f"Hasil retry OCR: **{result}**")
                if result == "ocr_done":
                    extract_result = process_document(document_id, doc["file_name"])
                    st.write(f"Hasil ekstraksi lanjutan: **{extract_result}**")
            else:
                st.error(
                    "File PDF asli tidak ditemukan di lokasi tersimpan "
                    f"({doc['local_path']}). Upload ulang lewat tab 'Upload & Proses'."
                )
        st.rerun()


def render_sync_tab():
    st.subheader("Sync ke Google Drive & Sheets")
    st.caption(
        "Upload PDF dokumen yang sudah 'Direview' ke Google Drive, lalu sync "
        "field-nya ke Google Sheets. Idempoten & append-only — aman dijalankan "
        "berkali-kali, tidak akan membuat duplikat atau menimpa baris lama."
    )

    with get_connection() as conn:
        counts = get_status_counts(conn)
    n_ready = counts.get("reviewed", 0)
    n_pending_sheet_sync = counts.get("uploaded_drive", 0)

    col1, col2 = st.columns(2)
    col1.metric("Siap di-sync (reviewed)", n_ready)
    col2.metric("Sudah di Drive, menunggu Sheets", n_pending_sheet_sync)

    if n_ready == 0 and n_pending_sheet_sync == 0:
        st.info("Tidak ada dokumen yang perlu di-sync saat ini.")
        return

    if st.button("☁️ Mulai Sync ke Drive & Sheets", type="primary"):
        try:
            validate_google_config()
        except ConfigError as exc:
            st.error(f"Konfigurasi Google belum lengkap: {exc}")
            st.caption(
                "Pastikan GOOGLE_APPLICATION_CREDENTIALS, GOOGLE_SHEET_ID, dan "
                "GOOGLE_DRIVE_FOLDER_ID sudah diset di environment variable."
            )
            return

        if not is_online():
            st.error(
                "Tidak ada koneksi internet (gagal konek ke sheets.googleapis.com). "
                "Dokumen tetap berstatus 'reviewed', coba lagi nanti."
            )
            return

        with st.spinner("Menyiapkan koneksi Google..."):
            creds = get_google_credentials()
            drive_service = get_drive_service(creds)
            sheets_client = get_sheets_client(creds)
            worksheet = get_worksheet(sheets_client)
            ensure_header(worksheet)

        with st.spinner(f"Upload {n_ready} PDF ke Google Drive..."):
            drive_results = upload_reviewed_documents_to_drive(drive_service)

        with st.spinner("Sync field ke Google Sheets..."):
            sync_results = sync_to_sheets(worksheet)

        st.success(
            f"✅ {drive_results['uploaded_drive']} PDF berhasil di-upload ke Drive, "
            f"{sync_results.get('synced', 0)} dokumen tersinkron ke Sheets "
            f"({sync_results.get('new_rows', 0)} baris baru, "
            f"{sync_results.get('already_existed', 0)} sudah ada sebelumnya)."
        )
        if drive_results["error"] > 0:
            st.warning(
                f"⚠️ {drive_results['error']} dokumen gagal di-upload — cek tab "
                "Ringkasan/Riwayat untuk detail error dan opsi retry."
            )
        st.rerun()


st.title("📄 OCR Batch Dokumen Dinas")

with get_connection() as conn:
    _counts = get_status_counts(conn)
st.caption(
    f"Total dokumen: {sum(_counts.values())} | "
    f"Menunggu review: {_counts.get('extracted', 0)} | "
    f"Error: {_counts.get('error', 0)}"
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["📤 Upload & Proses", "✅ Verifikasi & Edit", "☁️ Sync Drive & Sheets", "📊 Ringkasan/Riwayat"]
)
with tab1:
    render_upload_tab()
with tab2:
    render_review_tab()
with tab3:
    render_sync_tab()
with tab4:
    render_summary_tab()
