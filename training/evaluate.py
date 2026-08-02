"""Evaluasi model pada split test: akurasi + confusion matrix tiap kepala.

Contoh:
    python training/evaluate.py --data-dir data --model models/model.pt
"""

import argparse
import json
from pathlib import Path

import sys

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from app.labels import KATEGORI, UKURAN  # noqa: E402
from app.model import muat_model, transform_eval  # noqa: E402
from training.dataset import DatasetSampah, baca_manifest  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--model", default="models/model.pt")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    df = baca_manifest(data_dir / "labels.csv")

    split_file = Path(args.model).parent / "test_split.json"
    if split_file.exists():
        nama_test = set(json.loads(split_file.read_text(encoding="utf-8")))
        df = df[df["filename"].isin(nama_test)]
        print(f"Evaluasi pada {len(df)} foto split test.")
    else:
        print("test_split.json tak ditemukan — evaluasi pada SELURUH data.")

    model = muat_model(args.model, device="cpu")
    loader = DataLoader(DatasetSampah(df, data_dir / "images", transform_eval), batch_size=16)

    y_kat_true, y_kat_pred, y_uk_true, y_uk_pred = [], [], [], []
    with torch.no_grad():
        for x, y_kat, y_uk in loader:
            out_kat, out_uk = model(x)
            y_kat_true += y_kat.tolist()
            y_uk_true += y_uk.tolist()
            y_kat_pred += out_kat.argmax(1).tolist()
            y_uk_pred += out_uk.argmax(1).tolist()

    print("\n=== KATEGORI (organik/anorganik) ===")
    print(classification_report(y_kat_true, y_kat_pred, target_names=KATEGORI, zero_division=0))
    print("Confusion matrix (baris=asli, kolom=prediksi):")
    print(pd.DataFrame(confusion_matrix(y_kat_true, y_kat_pred), index=KATEGORI, columns=KATEGORI))

    print("\n=== UKURAN (kecil/sedang/besar) ===")
    print(classification_report(y_uk_true, y_uk_pred, target_names=UKURAN, zero_division=0))
    print("Confusion matrix (baris=asli, kolom=prediksi):")
    print(pd.DataFrame(confusion_matrix(y_uk_true, y_uk_pred), index=UKURAN, columns=UKURAN))


if __name__ == "__main__":
    main()
