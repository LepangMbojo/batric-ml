"""Latih model multi-task (kategori + ukuran).

Transfer learning bertahap:
  Tahap 1 — backbone dibekukan, latih hanya dua kepala (cepat, stabil).
  Tahap 2 — backbone di-unfreeze, fine-tune seluruh jaringan dengan lr kecil.

Loss = CrossEntropy(kategori) + CrossEntropy(ukuran). Bobot boleh disetel lewat
--bobot-ukuran kalau salah satu tugas lebih penting/lebih sulit.

Contoh:
    python training/train.py --data-dir data --epochs-head 5 --epochs-finetune 10
Hasil: models/model.pt (state_dict) + models/metrics.json
"""

import argparse
import json
from pathlib import Path

import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from app.model import KlasifikasiSampah, transform_eval, transform_train  # noqa: E402
from training.dataset import DatasetSampah, baca_manifest, split_stratified  # noqa: E402


def buat_loader(df, images_dir, transform, batch, shuffle):
    return DataLoader(
        DatasetSampah(df, images_dir, transform),
        batch_size=batch,
        shuffle=shuffle,
        num_workers=0,  # 0 = paling aman lintas OS (Windows); naikkan di Linux.
    )


def jalankan_epoch(model, loader, kriteria, device, bobot_ukuran, optimizer=None):
    latih = optimizer is not None
    model.train() if latih else model.eval()
    total_loss = benar_kat = benar_uk = n = 0
    with torch.set_grad_enabled(latih):
        for x, y_kat, y_uk in loader:
            x, y_kat, y_uk = x.to(device), y_kat.to(device), y_uk.to(device)
            out_kat, out_uk = model(x)
            loss = kriteria(out_kat, y_kat) + bobot_ukuran * kriteria(out_uk, y_uk)
            if latih:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            benar_kat += (out_kat.argmax(1) == y_kat).sum().item()
            benar_uk += (out_uk.argmax(1) == y_uk).sum().item()
            n += x.size(0)
    return total_loss / n, benar_kat / n, benar_uk / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--epochs-head", type=int, default=5)
    ap.add_argument("--epochs-finetune", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--bobot-ukuran", type=float, default=1.0)
    ap.add_argument("--out", default="models/model.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)
    df = baca_manifest(data_dir / "labels.csv")
    train_df, val_df, test_df = split_stratified(df)
    print(f"Data: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    images = data_dir / "images"
    train_loader = buat_loader(train_df, images, transform_train, args.batch, True)
    val_loader = buat_loader(val_df, images, transform_eval, args.batch, False)

    model = KlasifikasiSampah(pretrained=True).to(device)
    kriteria = nn.CrossEntropyLoss()

    def latih(fase, epochs, lr):
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        for ep in range(1, epochs + 1):
            tr = jalankan_epoch(model, train_loader, kriteria, device, args.bobot_ukuran, opt)
            va = jalankan_epoch(model, val_loader, kriteria, device, args.bobot_ukuran)
            print(
                f"[{fase} {ep}/{epochs}] "
                f"train loss={tr[0]:.3f} kat={tr[1]:.2f} uk={tr[2]:.2f} | "
                f"val loss={va[0]:.3f} kat={va[1]:.2f} uk={va[2]:.2f}"
            )

    # Tahap 1: kepala saja.
    model.bekukan_backbone(True)
    latih("head", args.epochs_head, lr=1e-3)
    # Tahap 2: fine-tune penuh, lr lebih kecil.
    model.bekukan_backbone(False)
    latih("finetune", args.epochs_finetune, lr=1e-4)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    print(f"Model tersimpan: {out}")

    # Simpan indeks split test untuk dievaluasi terpisah oleh evaluate.py.
    (out.parent / "test_split.json").write_text(
        json.dumps(test_df["filename"].tolist()), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
