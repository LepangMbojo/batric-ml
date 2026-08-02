"""Skema respons API — pydantic, tanpa dependensi torch."""

from pydantic import BaseModel, Field


class PrediksiResponse(BaseModel):
    kategori: str = Field(..., description="organik | anorganik")
    kategori_confidence: float = Field(..., ge=0, le=1)
    ukuran: str = Field(..., description="kecil | sedang | besar")
    ukuran_confidence: float = Field(..., ge=0, le=1)
    # true = prediksi masih dari model stub (dataset belum dilatih), jadi
    # frontend/analis tahu hasilnya belum bisa dipercaya.
    stub: bool = Field(..., description="true jika hasil dari model dummy")


class HealthResponse(BaseModel):
    status: str = "ok"
    stub: bool
