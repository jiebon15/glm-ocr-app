"""
Modul ekstraksi field terstruktur: teks OCR mentah -> 27 field sesuai skema
Google Sheets, memakai model text-only kedua (qwen2.5:7b) via Ollama dengan
constrained JSON output (parameter `format` berisi JSON Schema).
"""
import json
import logging

from ollama import Client

from app.config import EXTRACTION_MODEL_NAME, EXTRACTION_NUM_CTX

logger = logging.getLogger(__name__)

# Field yang diekstrak model (TIDAK termasuk: id, document_id,
# tanggal_diinput [auto], nama_file [dari nama file asli, bukan hasil LLM],
# drive_file_url/synced/synced_at [diisi Tahap 4]).
_STR_OR_NULL = {"type": ["string", "null"]}

EXTRACTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "nomor_surat": _STR_OR_NULL,
        "jenis_kegiatan": _STR_OR_NULL,
        "provinsi_lokasi": _STR_OR_NULL,
        "pelaksanaan": _STR_OR_NULL,
        "tanggal_surat": {
            **_STR_OR_NULL,
            "description": "Format YYYY-MM-DD jika bisa ditentukan, null jika tidak jelas.",
        },
        "jabatan_penandatangan": _STR_OR_NULL,
        "penandatangan": _STR_OR_NULL,
        "petugas1_nama": _STR_OR_NULL,
        "petugas1_nip": _STR_OR_NULL,
        "petugas1_no_reg": _STR_OR_NULL,
        "petugas2_nama": _STR_OR_NULL,
        "petugas2_nip": _STR_OR_NULL,
        "petugas2_no_reg": _STR_OR_NULL,
        "petugas3_nama": _STR_OR_NULL,
        "petugas3_nip": _STR_OR_NULL,
        "petugas3_no_reg": _STR_OR_NULL,
        "petugas4_nama": _STR_OR_NULL,
        "petugas4_nip": _STR_OR_NULL,
        "petugas4_no_reg": _STR_OR_NULL,
        "nama_upi": _STR_OR_NULL,
        "alamat_upi": _STR_OR_NULL,
        "jenis_produk_grade": _STR_OR_NULL,
        "no_reg": {
            **_STR_OR_NULL,
            "description": (
                "Nomor registrasi DOKUMEN secara keseluruhan, "
                "BUKAN nomor registrasi profesi petugas."
            ),
        },
    },
    # `required` di JSON Schema Ollama berarti key harus MUNCUL di output
    # (nilainya tetap boleh null berkat type union di atas) — ini memaksa
    # model konsisten mengisi semua 22 field, bukan berarti tidak boleh null.
    "required": [
        "nomor_surat", "jenis_kegiatan", "provinsi_lokasi", "pelaksanaan",
        "tanggal_surat", "jabatan_penandatangan", "penandatangan",
        "petugas1_nama", "petugas1_nip", "petugas1_no_reg",
        "petugas2_nama", "petugas2_nip", "petugas2_no_reg",
        "petugas3_nama", "petugas3_nip", "petugas3_no_reg",
        "petugas4_nama", "petugas4_nip", "petugas4_no_reg",
        "nama_upi", "alamat_upi", "jenis_produk_grade", "no_reg",
    ],
}

EXTRACTION_SYSTEM_PROMPT = """\
Anda adalah asisten ekstraksi data dari surat dinas hasil OCR. Baca teks
surat di bawah dan keluarkan HANYA JSON sesuai skema yang diberikan.

ATURAN PENTING:
1. Isi null untuk field apa pun yang TIDAK disebutkan secara eksplisit di
   teks. JANGAN mengarang nama, NIP, nomor, atau tanggal yang tidak ada.
2. Petugas 2, 3, dan 4 boleh seluruhnya null jika surat hanya melibatkan
   1-3 petugas (lihat daftar nama pada bagian penugasan/lampiran surat).
3. Ada DUA jenis "No. Reg" yang BERBEDA — jangan sampai tertukar:
   a. `no_reg` (di level dokumen) = nomor registrasi SURAT/DOKUMEN itu
      sendiri, biasanya muncul di kop surat atau nomor administrasi surat.
   b. `petugasN_no_reg` = nomor registrasi PROFESI petugas yang
      bersangkutan (mis. nomor register PPC/inspektur), biasanya muncul
      di daftar nama petugas/pelaksana, sering berdampingan dengan NIP
      petugas tsb.

CONTOH (few-shot) supaya tidak tertukar:
---
Cuplikan surat:
  "Nomor: 556/UPT.SBY/OC/VII/2026
   ...
   Petugas Pelaksana:
   1. Budi Santoso, NIP 198501012010011001, No. Reg PPC.0231
   2. Siti Aminah, NIP 199002022012022002, No. Reg PPC.0455"

Ekstraksi yang BENAR:
  no_reg = "556/UPT.SBY/OC/VII/2026"   <- nomor DOKUMEN (dari kop surat)
  petugas1_nama = "Budi Santoso"
  petugas1_nip = "198501012010011001"
  petugas1_no_reg = "PPC.0231"          <- nomor REGISTRASI PROFESI petugas 1
  petugas2_nama = "Siti Aminah"
  petugas2_nip = "199002022012022002"
  petugas2_no_reg = "PPC.0455"
  petugas3_nama = null
  petugas3_nip = null
  petugas3_no_reg = null
---

4. Nama file dokumen akan diberikan sebagai KONTEKS TAMBAHAN (bukan isi
   surat). Nama file sering memuat nama UPI (mis. "1407 - Official
   Control GABUNGAN SAMUDERA INTERNASIONAL, PT.pdf" -> nama UPI-nya
   "GABUNGAN SAMUDERA INTERNASIONAL, PT"). Gunakan ini sebagai REFERENSI
   SILANG untuk membantu menentukan `nama_upi` yang benar, terutama jika
   teks OCR pada bagian nama UPI meragukan, terpotong, atau ada salah
   ketik hasil OCR. TETAP prioritaskan isi surat sebagai sumber utama;
   nama file hanya alat bantu verifikasi, bukan pengganti jika isi surat
   jelas menyebutkan UPI yang berbeda.

Sekarang ekstrak dari teks surat berikut. Balas HANYA dengan objek JSON,
tanpa teks pembuka, penutup, atau markdown code fence.
"""


def get_extraction_client() -> Client:
    from app.config import OLLAMA_HOST
    return Client(host=OLLAMA_HOST)


def extract_fields(ocr_text: str, client: Client, file_name: str | None = None) -> dict:
    """Kirim teks OCR gabungan (+ nama file sebagai konteks referensi
    silang nama UPI) ke model ekstraksi, kembalikan dict field.

    Melempar ValueError jika model gagal menghasilkan JSON valid sesuai
    skema (jarang terjadi karena constrained output, tapi tetap dijaga).
    """
    user_content = ocr_text
    if file_name:
        user_content = (
            f"[Nama file dokumen (konteks referensi, lihat aturan #4): {file_name}]\n\n"
            f"{ocr_text}"
        )

    response = client.chat(
        model=EXTRACTION_MODEL_NAME,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        format=EXTRACTION_JSON_SCHEMA,
        options={"num_ctx": EXTRACTION_NUM_CTX, "temperature": 0},
    )
    raw_content = response["message"]["content"]
    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Model ekstraksi tidak mengembalikan JSON valid: {exc}\n"
            f"Isi mentah: {raw_content[:500]}"
        ) from exc

    # Validasi ringan: pastikan semua key yang diharapkan ada, isi None
    # untuk key yang hilang (jaga-jaga meski constrained output seharusnya
    # sudah menjamin ini).
    for key in EXTRACTION_JSON_SCHEMA["required"]:
        parsed.setdefault(key, None)

    return parsed
