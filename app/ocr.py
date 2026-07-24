"""
Modul OCR: PDF -> gambar per halaman -> teks mentah via model GLM-OCR (Ollama).
"""
import io
import logging
from pathlib import Path
from typing import Iterator

from ollama import Client
from pdf2image import convert_from_path
from PIL import Image

from app.config import OCR_MODEL_NAME, OLLAMA_HOST, OCR_NUM_CTX

logger = logging.getLogger(__name__)

# Prompt instruksi ke GLM-OCR: minta transkripsi apa adanya, tanpa
# interpretasi/ringkasan, karena hasil ini akan diproses lagi oleh model
# ekstraksi field di Tahap 2.
OCR_PROMPT = (
    "Transkripsikan seluruh teks yang terlihat pada gambar dokumen ini "
    "apa adanya, dalam format markdown sederhana. Jangan meringkas, "
    "menerjemahkan, atau menambahkan interpretasi. Jika ada tabel, "
    "pertahankan strukturnya semirip mungkin."
)


def pdf_to_images(pdf_path: Path, dpi: int = 300) -> Iterator[Image.Image]:
    """Konversi tiap halaman PDF menjadi PIL Image (butuh poppler-utils)."""
    images = convert_from_path(str(pdf_path), dpi=dpi)
    for image in images:
        yield image


def _image_to_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def ocr_image(image: Image.Image, client: Client) -> str:
    """Kirim satu gambar halaman ke model GLM-OCR via Ollama, kembalikan teks."""
    response = client.chat(
        model=OCR_MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": OCR_PROMPT,
                "images": [_image_to_bytes(image)],
            }
        ],
        # Default context window Ollama (4096) sering tidak cukup untuk
        # gambar hasil scan dokumen + prompt. Dibuat bisa diatur lewat
        # env var OCR_NUM_CTX karena jumlah token gambar tergantung
        # resolusi/DPI dan model vision encoder yang dipakai.
        options={"num_ctx": OCR_NUM_CTX},
    )
    return response["message"]["content"]


def get_ollama_client() -> Client:
    return Client(host=OLLAMA_HOST)
