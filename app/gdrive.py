"""
Modul upload PDF ke Google Drive (Tahap 4), pakai Service Account yang
sama dengan gsheets.py. Folder tujuan FLAT (tanpa subfolder/kategori).
"""
import logging
import socket
from pathlib import Path

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from app.config import GOOGLE_APPLICATION_CREDENTIALS, GOOGLE_DRIVE_FOLDER_ID

logger = logging.getLogger(__name__)

# Scope gabungan untuk Drive + Sheets, dipakai dari satu file kredensial
# Service Account yang sama (sesuai desain proyek).
GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
]


def is_online(host: str = "sheets.googleapis.com", port: int = 443, timeout: float = 3.0) -> bool:
    """Cek koneksi internet sederhana sebelum mencoba sync (sesuai desain:
    aplikasi harus tetap bisa dipakai offline, sync ditunda kalau
    tidak online)."""
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except OSError:
        return False


def get_google_credentials() -> Credentials:
    return Credentials.from_service_account_file(
        GOOGLE_APPLICATION_CREDENTIALS, scopes=GOOGLE_SCOPES
    )


def get_drive_service(creds: Credentials):
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def upload_pdf_to_drive(service, local_path: Path, file_name: str) -> str:
    """Upload satu file PDF ke folder Drive tujuan (flat, tanpa subfolder).
    Return webViewLink (URL yang disimpan ke kolom Drive File URL)."""
    file_metadata = {"name": file_name, "parents": [GOOGLE_DRIVE_FOLDER_ID]}
    media = MediaFileUpload(str(local_path), mimetype="application/pdf", resumable=True)
    uploaded = (
        service.files()
        .create(body=file_metadata, media_body=media, fields="id, webViewLink")
        .execute()
    )
    logger.info("Upload Drive sukses: %s -> file_id=%s", file_name, uploaded["id"])
    return uploaded["webViewLink"]
