"""Dataset multi-label dari manifest CSV.

Format labels.csv (header wajib):
    filename,kategori,ukuran
    0001.jpg,anorganik,sedang
    0002.jpg,organik,kecil

Tiap foto punya DUA label (kategori & ukuran), jadi dipakai manifest CSV,
bukan struktur folder-per-kelas (yang hanya cocok untuk satu dimensi label).
"""

from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from app.labels import KATEGORI, UKURAN  # noqa: E402


class DatasetSampah(Dataset):
    def __init__(self, df: pd.DataFrame, images_dir: Path, transform):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        baris = self.df.iloc[idx]
        img = Image.open(self.images_dir / baris["filename"]).convert("RGB")
        x = self.transform(img)
        y_kat = KATEGORI.index(baris["kategori"])
        y_uk = UKURAN.index(baris["ukuran"])
        return x, y_kat, y_uk


def baca_manifest(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    wajib = {"filename", "kategori", "ukuran"}
    if not wajib.issubset(df.columns):
        raise ValueError(f"labels.csv harus punya kolom: {sorted(wajib)}")
    # Validasi nilai label agar salah ketik ketahuan lebih awal.
    bad_kat = set(df["kategori"]) - set(KATEGORI)
    bad_uk = set(df["ukuran"]) - set(UKURAN)
    if bad_kat:
        raise ValueError(f"Nilai kategori tak dikenal: {bad_kat}. Harus {KATEGORI}")
    if bad_uk:
        raise ValueError(f"Nilai ukuran tak dikenal: {bad_uk}. Harus {UKURAN}")
    return df


def split_stratified(df: pd.DataFrame, val_size=0.15, test_size=0.15, seed=42):
    """Split train/val/test.

    Stratifikasi pakai kombinasi kategori+ukuran supaya proporsi tiap kelas
    seimbang di tiap split. Kalau data terlalu sedikit untuk stratifikasi, jatuh
    ke split acak biasa.
    """
    strata = df["kategori"] + "_" + df["ukuran"]
    try:
        train, sisa = train_test_split(
            df, test_size=val_size + test_size, random_state=seed, stratify=strata
        )
        strata_sisa = sisa["kategori"] + "_" + sisa["ukuran"]
        rel_test = test_size / (val_size + test_size)
        val, test = train_test_split(
            sisa, test_size=rel_test, random_state=seed, stratify=strata_sisa
        )
    except ValueError:
        # Kelas terlalu kecil untuk stratify — split acak biasa.
        train, sisa = train_test_split(df, test_size=val_size + test_size, random_state=seed)
        val, test = train_test_split(sisa, test_size=test_size / (val_size + test_size), random_state=seed)
    return train, val, test
