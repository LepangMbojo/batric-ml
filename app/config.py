"""Konfigurasi service dari environment variable.

Semua pengaturan runtime dibaca di satu tempat supaya mudah dilacak.
"""

import os

# Muat file .env (kalau ada) SEBELUM os.getenv dibaca. uvicorn tidak melakukan
# ini sendiri. Aman kalau .env tak ada (default di bawah tetap dipakai) atau
# python-dotenv belum terpasang (fallback ke env var proses).
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def _as_bool(nilai: str, default: bool) -> bool:
    if nilai is None:
        return default
    return nilai.strip().lower() in {"1", "true", "yes", "on"}


# STUB_MODE=true → jangan load bobot / import torch, kembalikan prediksi dummy.
# Default true supaya `git clone` + jalankan langsung bisa tanpa dataset/torch.
STUB_MODE = _as_bool(os.getenv("STUB_MODE"), default=True)

# Lokasi bobot terlatih (dipakai hanya saat STUB_MODE=false).
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pt")

# Origin yang boleh memanggil (PWA warga). Dipisah koma.
# Default mencakup port dev Vite umum.
ML_ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "ML_ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if o.strip()
]

# Batas ukuran unggahan (byte) — samakan dengan aturan foto laporan (5 MB).
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))
