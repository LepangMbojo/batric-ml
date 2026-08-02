"""Arsitektur model multi-task + transform gambar.

Modul ini meng-import torch, jadi HANYA diimpor saat STUB_MODE=false (inferensi
model asli) atau saat training. Selama mode stub, torch tidak perlu ter-install.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

from .labels import KATEGORI, UKURAN

# Ukuran input & normalisasi standar ImageNet (wajib sama saat latih & inferensi).
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

# Transform untuk INFERENSI / validasi (tanpa augmentasi acak).
transform_eval = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ]
)

# Transform untuk TRAINING (dengan augmentasi — foto sampah sangat bervariasi).
transform_train = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ]
)


class KlasifikasiSampah(nn.Module):
    """Satu backbone MobileNetV3-Large + dua kepala klasifikasi (multi-task).

    Kepala A → kategori (organik/anorganik), Kepala B → ukuran (kecil/sedang/besar).
    Fitur backbone dipakai bersama sehingga model tetap ringan.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_large(weights=weights)

        # MobileNetV3: .features (ekstraktor) + .avgpool + .classifier.
        # Kita pakai .features + pooling sendiri, lalu dua kepala terpisah.
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        in_features = backbone.classifier[0].in_features  # 960 untuk V3-Large

        def buat_kepala(jumlah_kelas: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_features, 256),
                nn.Hardswish(),
                nn.Dropout(0.2),
                nn.Linear(256, jumlah_kelas),
            )

        self.kepala_kategori = buat_kepala(len(KATEGORI))
        self.kepala_ukuran = buat_kepala(len(UKURAN))

    def forward(self, x):
        f = self.pool(self.features(x)).flatten(1)
        return self.kepala_kategori(f), self.kepala_ukuran(f)

    def bekukan_backbone(self, beku: bool = True):
        """Freeze/unfreeze backbone untuk transfer learning bertahap."""
        for p in self.features.parameters():
            p.requires_grad = not beku


def muat_model(model_path: str, device: str = "cpu") -> KlasifikasiSampah:
    """Bangun model & muat bobot terlatih untuk inferensi."""
    model = KlasifikasiSampah(pretrained=False)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
