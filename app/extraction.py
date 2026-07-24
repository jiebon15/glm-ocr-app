"""
Modul ekstraksi field terstruktur: teks OCR mentah -> 23 field via model
text-only kedua (qwen2.5:7b, Ollama, constrained JSON output).

ARSITEKTUR DUA-PANGGILAN:
Field dikelompokkan jadi dua, karena sumber teksnya berbeda:

1. `ADMIN_FIELDS` — field administratif yang selalu ada di 1-2 halaman
   pertama surat (kop surat & blok tanda tangan): nomor_surat,
   jenis_kegiatan, provinsi_lokasi, pelaksanaan, tanggal_surat,
   jabatan_penandatangan, penandatangan.
   -> diekstrak dari teks OCR halaman 1-2 SAJA (dibatasi via
      EXTRACTION_MAX_PAGES), supaya tidak "tersesat" ke lampiran.

2. `FULL_DOC_FIELDS` — field yang bisa muncul di mana saja termasuk
   lampiran: petugas1-4 (nama/nip/no_reg), nama_upi, alamat_upi,
   jenis_produk_grade, no_reg (nomor registrasi dokumen).
   -> diekstrak dari teks OCR SELURUH halaman dokumen.

Hasil kedua panggilan digabung (dict merge) sebelum disimpan ke DB.
"""
import json
import logging

from ollama import Client

from app.config import EXTRACTION_FULL_NUM_CTX, EXTRACTION_MODEL_NAME, EXTRACTION_NUM_CTX

logger = logging.getLogger(__name__)

_STR_OR_NULL = {"type": ["string", "null"]}

# --- Kelompok 1: field administratif, sumber = halaman 1-2 ---
ADMIN_FIELDS = [
    "nomor_surat", "jenis_kegiatan", "provinsi_lokasi", "pelaksanaan",
    "tanggal_surat", "jabatan_penandatangan", "penandatangan",
]

# --- Kelompok 2: field lain, sumber = seluruh dokumen (termasuk lampiran) ---
FULL_DOC_FIELDS = [
    "petugas1_nama", "petugas1_nip", "petugas1_no_reg",
    "petugas2_nama", "petugas2_nip", "petugas2_no_reg",
    "petugas3_nama", "petugas3_nip", "petugas3_no_reg",
    "petugas4_nama", "petugas4_nip", "petugas4_no_reg",
    "nama_upi", "alamat_upi", "jenis_produk_grade", "no_reg",
]

# Deskripsi khusus untuk field tertentu (dipakai di JSON schema)
_FIELD_DESCRIPTIONS = {
    "tanggal_surat": "Format YYYY-MM-DD jika bisa ditentukan, null jika tidak jelas.",
    "no_reg": (
        "Nomor registrasi DOKUMEN secara keseluruhan, "
        "BUKAN nomor registrasi profesi petugas."
    ),
}


def _build_schema(field_names: list[str]) -> dict:
    """Bangun JSON Schema (untuk parameter `format` Ollama) yang hanya
    berisi field_names, semua bertipe string-atau-null."""
    properties = {}
    for name in field_names:
        prop = dict(_STR_OR_NULL)
        if name in _FIELD_DESCRIPTIONS:
            prop["description"] = _FIELD_DESCRIPTIONS[name]
        properties[name] = prop
    return {
        "type": "object",
        "properties": properties,
        # `required` di sini berarti key harus MUNCUL di output (nilainya
        # tetap boleh null berkat type union) — bukan berarti wajib terisi.
        "required": list(field_names),
    }


ADMIN_JSON_SCHEMA = _build_schema(ADMIN_FIELDS)
FULL_DOC_JSON_SCHEMA = _build_schema(FULL_DOC_FIELDS)

# Untuk kompatibilitas/inspeksi: gabungan skema semua 23 field.
EXTRACTION_JSON_SCHEMA = _build_schema(ADMIN_FIELDS + FULL_DOC_FIELDS)


ADMIN_SYSTEM_PROMPT = """\
Anda adalah asisten ekstraksi data dari surat dinas hasil OCR. Anda hanya
diberikan 1-2 halaman PERTAMA surat (kop surat & blok tanda tangan) —
bagian ini biasanya memuat nomor surat, jenis kegiatan, lokasi, tanggal,
dan jabatan/nama penandatangan.

ATURAN:
1. Isi null untuk field apa pun yang TIDAK disebutkan secara eksplisit di
   teks. JANGAN mengarang nomor, tanggal, atau nama yang tidak ada.
2. `jabatan_penandatangan` = jabatan/posisi resmi orang yang menandatangani
   surat (mis. "Kepala UPT"). `penandatangan` = nama orang tersebut.
   Keduanya biasanya berdampingan di blok tanda tangan akhir surat.
3. `pelaksanaan` = keterangan waktu/periode pelaksanaan kegiatan
   (mis. rentang tanggal kegiatan), BUKAN tanggal surat itu sendiri.
4. `tanggal_surat` SERING TIDAK berlabel "Tanggal:" secara eksplisit.
   Pola paling umum: baris "<Kota>, <tanggal>" yang berdiri sendiri
   TEPAT DI ATAS blok jabatan_penandatangan/penandatangan, contoh:

     Surabaya, 24 Juli 2026
     Kepala UPT
     [ttd]
     Ahmad Fauzi

   Di sini tanggal_surat = "2026-07-24" (bukan null), diambil dari baris
   kota+tanggal itu meski tidak ada kata "Tanggal:". Konversi nama bulan
   Indonesia (Januari-Desember) ke angka 01-12 untuk format YYYY-MM-DD.
   Kalau tahun/bulan/tanggal tidak lengkap atau tidak jelas, baru isi null.

Balas HANYA dengan objek JSON sesuai skema, tanpa teks pembuka, penutup,
atau markdown code fence.
"""

FULL_DOC_SYSTEM_PROMPT = """\
Anda adalah asisten ekstraksi data dari surat dinas hasil OCR. Anda
diberikan teks OCR SELURUH halaman dokumen (surat utama + lampiran, jika
ada). Field yang diminta di sini bisa muncul di halaman mana pun,
termasuk lampiran (mis. daftar petugas pelaksana sering ada di lampiran).

ATURAN PENTING:
1. Isi null untuk field apa pun yang TIDAK disebutkan secara eksplisit di
   teks. JANGAN mengarang nama, NIP, atau nomor yang tidak ada.
2. Petugas 2, 3, dan 4 boleh seluruhnya null jika surat hanya melibatkan
   1-3 petugas.
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

Balas HANYA dengan objek JSON sesuai skema, tanpa teks pembuka, penutup,
atau markdown code fence.
"""


def get_extraction_client() -> Client:
    from app.config import OLLAMA_HOST
    return Client(host=OLLAMA_HOST)


def _call_extraction_model(
    client: Client,
    system_prompt: str,
    user_content: str,
    json_schema: dict,
    num_ctx: int,
) -> dict:
    response = client.chat(
        model=EXTRACTION_MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        format=json_schema,
        options={"num_ctx": num_ctx, "temperature": 0},
    )
    raw_content = response["message"]["content"]
    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Model ekstraksi tidak mengembalikan JSON valid: {exc}\n"
            f"Isi mentah: {raw_content[:500]}"
        ) from exc

    for key in json_schema["required"]:
        parsed.setdefault(key, None)
    return parsed


def extract_fields(
    admin_text: str,
    full_text: str,
    client: Client,
    file_name: str | None = None,
) -> dict:
    """Jalankan dua panggilan model ekstraksi dan gabungkan hasilnya.

    `admin_text`: teks OCR halaman 1-2 (untuk field administratif).
    `full_text`: teks OCR seluruh halaman (untuk petugas/UPI/no_reg).

    Melempar ValueError jika salah satu panggilan gagal menghasilkan JSON
    valid sesuai skema.
    """
    admin_result = _call_extraction_model(
        client, ADMIN_SYSTEM_PROMPT, admin_text, ADMIN_JSON_SCHEMA, EXTRACTION_NUM_CTX,
    )

    full_user_content = full_text
    if file_name:
        full_user_content = (
            f"[Nama file dokumen (konteks referensi, lihat aturan #4): {file_name}]\n\n"
            f"{full_text}"
        )
    full_result = _call_extraction_model(
        client, FULL_DOC_SYSTEM_PROMPT, full_user_content, FULL_DOC_JSON_SCHEMA,
        EXTRACTION_FULL_NUM_CTX,
    )

    return {**admin_result, **full_result}
