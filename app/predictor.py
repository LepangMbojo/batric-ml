"""Inferensi satu foto → prediksi kategori & ukuran.

Punya dua mode:
  - STUB  : tanpa torch, kembalikan prediksi dummy deterministik (Fase A).
  - MODEL : muat bobot terlatih & jalankan CNN sungguhan (Fase B).

Torch hanya di-import di dalam metode _muat_model_asli(), jadi mode stub tidak
membutuhkan torch/torchvision ter-install sama sekali.
"""

import hashlib
import io

from PIL import Image, UnidentifiedImageError

from . import config
from .labels import KATEGORI, UKURAN


class FotoTidakValid(Exception):
    """Dilempar saat byte yang diunggah bukan gambar yang bisa dibaca."""


class Predictor:
    def __init__(self):
        self.stub = config.STUB_MODE
        self._model = None
        self._transform = None
        self._torch = None
        if not self.stub:
            self._muat_model_asli()

    def _muat_model_asli(self):
        # Import di sini (bukan di atas file) supaya mode stub bebas torch.
        import torch

        from .model import muat_model, transform_eval

        self._torch = torch
        self._transform = transform_eval
        self._model = muat_model(config.MODEL_PATH, device="cpu")

    def _decode(self, data: bytes) -> Image.Image:
        if len(data) > config.MAX_UPLOAD_BYTES:
            raise FotoTidakValid("Ukuran foto melebihi batas maksimal 5 MB.")
        try:
            img = Image.open(io.BytesIO(data))
            img.verify()  # cek integritas tanpa full-decode
            # verify() membuat objek tak terpakai lagi — buka ulang untuk dipakai.
            return Image.open(io.BytesIO(data)).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            raise FotoTidakValid("File yang diunggah bukan gambar yang valid.") from e

    def predict(self, data: bytes) -> dict:
        img = self._decode(data)
        if self.stub:
            return self._predict_stub(data)
        return self._predict_model(img)

    def _predict_stub(self, data: bytes) -> dict:
        """Prediksi dummy DETERMINISTIK berdasar hash foto.

        Deterministik (bukan acak murni) supaya foto sama selalu memberi hasil
        sama — memudahkan uji integrasi. Confidence sengaja rendah sebagai
        penanda bahwa ini belum model asli.
        """
        h = hashlib.sha256(data).digest()
        kategori = KATEGORI[h[0] % len(KATEGORI)]
        ukuran = UKURAN[h[1] % len(UKURAN)]
        return {
            "kategori": kategori,
            "kategori_confidence": 0.30 + (h[2] % 20) / 100,  # 0.30–0.49
            "ukuran": ukuran,
            "ukuran_confidence": 0.30 + (h[3] % 20) / 100,
            "stub": True,
        }

    def _predict_model(self, img: Image.Image) -> dict:
        torch = self._torch
        x = self._transform(img).unsqueeze(0)  # tambah dimensi batch
        with torch.no_grad():
            logit_kat, logit_uk = self._model(x)
            prob_kat = torch.softmax(logit_kat, dim=1)[0]
            prob_uk = torch.softmax(logit_uk, dim=1)[0]
        i_kat = int(prob_kat.argmax())
        i_uk = int(prob_uk.argmax())
        return {
            "kategori": KATEGORI[i_kat],
            "kategori_confidence": float(prob_kat[i_kat]),
            "ukuran": UKURAN[i_uk],
            "ukuran_confidence": float(prob_uk[i_uk]),
            "stub": False,
        }
