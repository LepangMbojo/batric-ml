"""Service FastAPI: klasifikasi foto sampah.

Endpoint:
  GET  /health   → status service + apakah masih mode stub
  POST /predict  → terima foto (multipart), balikan kategori + ukuran

Dipanggil LANGSUNG dari PWA warga (pola sama dengan OSRM). Backend Laravel tidak
memanggil service ini — prediksi hanya pre-fill form yang bisa dikoreksi warga.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .predictor import FotoTidakValid, Predictor
from .schema import HealthResponse, PrediksiResponse

# Predictor di-load SEKALI saat startup (mahal), lalu dipakai ulang tiap request.
_predictor: Predictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor
    _predictor = Predictor()
    yield
    _predictor = None


app = FastAPI(title="BATRIC ML — Klasifikasi Sampah", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ML_ALLOWED_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", stub=config.STUB_MODE)


@app.post("/predict", response_model=PrediksiResponse)
async def predict(foto: UploadFile = File(...)):
    data = await foto.read()
    if not data:
        # 422: konsisten dengan gaya error validasi backend Laravel.
        raise HTTPException(status_code=422, detail="Foto tidak boleh kosong.")
    try:
        hasil = _predictor.predict(data)
    except FotoTidakValid as e:
        raise HTTPException(status_code=422, detail=str(e))
    return PrediksiResponse(**hasil)
