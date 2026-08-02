# BATRIC ML — Klasifikasi Sampah

Service inferensi: dari **foto sampah** memprediksi **kategori**
(organik/anorganik) dan **ukuran** (kecil/sedang/besar). Dipanggil langsung dari
PWA warga untuk *pre-fill* form lapor (warga tetap bisa mengoreksi).

Model: satu backbone MobileNetV3-Large (pretrained ImageNet) + dua kepala
klasifikasi (multi-task), PyTorch, disajikan via FastAPI.

## Dua mode

| Mode | STUB_MODE | Butuh torch? | Butuh dataset? | Kegunaan |
|------|-----------|--------------|----------------|----------|
| Stub | `true` (default) | ❌ | ❌ | Uji integrasi frontend↔backend (Fase A) |
| Model asli | `false` | ✅ | ✅ (sudah dilatih) | Produksi (Fase B) |

## Menjalankan (mode stub — tanpa dataset/torch)

```bash
cd batric-ml
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash
pip install fastapi "uvicorn[standard]" python-multipart pillow
cp .env.example .env                                    # STUB_MODE=true
uvicorn app.main:app --port 8001 --reload
```

Uji:

```bash
curl http://localhost:8001/health                       # {"status":"ok","stub":true}
curl -F "foto=@../batric-backend/test.jpg" http://localhost:8001/predict
```

## Melatih model asli (Fase B — setelah dataset ada)

1. Isi dataset:
   ```
   data/images/0001.jpg, 0002.jpg, ...
   data/labels.csv   # kolom: filename,kategori,ukuran
   ```
2. Install penuh & latih:
   ```bash
   pip install -r requirements.txt
   python training/train.py --data-dir data --epochs-head 5 --epochs-finetune 10
   python training/evaluate.py --data-dir data --model models/model.pt
   ```
3. Jalankan dengan model asli: set `STUB_MODE=false` di `.env`, lalu `uvicorn ...`.

## Kontrak API

`POST /predict` (multipart, field `foto`) →

```json
{ "kategori": "anorganik", "kategori_confidence": 0.93,
  "ukuran": "sedang", "ukuran_confidence": 0.71, "stub": false }
```

Error foto tak valid / kosong / > 5 MB → HTTP 422 (pesan Bahasa Indonesia).
