"""Sumber tunggal daftar label — dipakai bersama oleh service & training.

Urutan list = urutan indeks kelas pada output model (argmax mengembalikan
indeks, lalu dipetakan balik ke string lewat list ini). JANGAN mengubah urutan
setelah model dilatih, karena akan menggeser arti tiap indeks.
"""

KATEGORI = ["organik", "anorganik"]      # kepala A → 2 kelas
UKURAN = ["kecil", "sedang", "besar"]    # kepala B → 3 kelas
