# Import Library
import os
import time
import uuid
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from src import config
from src.logger import ambil_logger

# Inisialisasi Logger Terstruktur
log = ambil_logger()

# "auto" = Coba Registry Dulu, Kalau Gagal Pakai File Hasil Ekspor Lokal.
SUMBER_MODEL = os.environ.get("SUMBER_MODEL", "auto")

# Menentukan Metrik untuk Pengaturan Instrumen Prometheus 
METRIK_REQUEST = Counter(
    "prediksi_total", "Jumlah request prediksi yang masuk")
METRIK_ERROR = Counter(
    "prediksi_error_total", "Jumlah request yang gagal")
METRIK_LATENSI = Histogram(
    "prediksi_latency_seconds", "Lama waktu memproses satu prediksi",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0))
METRIK_HARGA = Histogram(
    "harga_prediksi_usd", "Sebaran harga prediksi dalam USD",
    buckets=(50000, 100000, 150000, 200000, 250000, 300000, 400000, 500000))
METRIK_VERSI = Gauge(
    "model_versi_aktif", "Versi model yang sedang dipakai")
METRIK_GRLIVAREA_RATA2 = Gauge(
    "input_grlivarea_rata2", "Rata-rata GrLivArea dari 100 request terakhir")
METRIK_DRIFT = Gauge(
    "skor_drift_grlivarea", "Seberapa jauh input bergeser dari data training")

# Baseline Nilai Rata-rata "GrLivArea" Dari Data Train untuk Deteksi Data Drift
GRLIVAREA_RATA2_TRAINING = 1515.0

# Membuat Object State Global
kondisi = {
    "model": None,
    "versi": "belum-dimuat",
    "sumber": "-",
    "grlivarea_terakhir": [],
}

# Membuat Function Pemuatan Artefak Model ke Dalam Memori
def muat_model():
    if SUMBER_MODEL in ("auto", "registry"):
        try:
            import mlflow
            import mlflow.sklearn
            from mlflow import MlflowClient

            # Hubungkan ke MLflow Tracking Server
            mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
            client = MlflowClient()
            versi = client.get_model_version_by_alias(
                config.NAMA_MODEL, config.ALIAS_PRODUKSI)
            alamat = "models:/" + config.NAMA_MODEL + "@" + config.ALIAS_PRODUKSI
            
            # Memuat Model dari Registry
            kondisi["model"] = mlflow.sklearn.load_model(alamat)
            kondisi["versi"] = str(versi.version)
            kondisi["sumber"] = "mlflow-registry"
            try:
                METRIK_VERSI.set(float(versi.version))
            except ValueError:
                pass
            log.info("model_dimuat", extra={
                "sumber": "mlflow-registry", "versi": kondisi["versi"]})
            return
        except Exception as salah:
            if SUMBER_MODEL == "registry":
                raise
            log.warning("registry_tidak_bisa_dipakai", extra={"pesan": str(salah)[:120]})

    # Fallback: Memuat File "model.joblib" Ekspor Lokal
    berkas = os.path.join(config.FOLDER_MODEL_EKSPOR, "model.joblib")
    if not os.path.exists(berkas):
        raise FileNotFoundError(f"File model tidak ditemukan di: {berkas}")

    kondisi["model"] = joblib.load(berkas)
    berkas_versi = os.path.join(config.FOLDER_MODEL_EKSPOR, "VERSION")
    if os.path.exists(berkas_versi):
        kondisi["versi"] = open(berkas_versi).read().strip()
    kondisi["sumber"] = "file-ekspor"
    try:
        METRIK_VERSI.set(float(kondisi["versi"]))
    except ValueError:
        pass
    log.info("model_dimuat", extra={"sumber": "file-ekspor", "versi": kondisi["versi"]})


# Pengelola Siklus Hidup (Lifespan) Aplikasi FastAPI
@asynccontextmanager
async def saat_hidup(app: FastAPI):
    muat_model()
    yield
    log.info("server_berhenti")


# Inisialisasi Instance FastAPI
app = FastAPI(
    title="API Prediksi Harga Rumah Kaggle",
    description="MLOps FastAPI + MLflow Model Registry + Prometheus Metrics",
    version="1.0.0",
    lifespan=saat_hidup,
)


# Membuat Skema Pydantic untuk Validasi Tipe Data dan Batasan Nilai Input Feature
class RumahRequest(BaseModel):
    MSSubClass: str = Field(..., json_schema_extra={"example": "60"})
    LotArea: int = Field(..., ge=0, json_schema_extra={"example": 8450})
    OverallQual: int = Field(..., ge=1, le=10, json_schema_extra={"example": 7})
    OverallCond: int = Field(..., ge=1, le=10, json_schema_extra={"example": 5})
    YearBuilt: int = Field(..., ge=1800, le=2030, json_schema_extra={"example": 2003})
    YearRemodAdd: int = Field(..., ge=1800, le=2030, json_schema_extra={"example": 2003})
    TotalBsmtSF: float = Field(..., ge=0, json_schema_extra={"example": 856.0})
    first_flr_sf: float = Field(..., alias="1stFlrSF", ge=0, json_schema_extra={"example": 856.0})
    second_flr_sf: float = Field(..., alias="2ndFlrSF", ge=0, json_schema_extra={"example": 854.0})
    GrLivArea: float = Field(..., ge=0, json_schema_extra={"example": 1710.0})
    FullBath: int = Field(..., ge=0, json_schema_extra={"example": 2})
    HalfBath: int = Field(..., ge=0, json_schema_extra={"example": 1})
    BedroomAbvGr: int = Field(..., ge=0, json_schema_extra={"example": 3})
    TotRmsAbvGrd: int = Field(..., ge=0, json_schema_extra={"example": 8})
    Fireplaces: int = Field(..., ge=0, json_schema_extra={"example": 0})
    GarageCars: int = Field(..., ge=0, json_schema_extra={"example": 2})
    GarageArea: float = Field(..., ge=0, json_schema_extra={"example": 548.0})
    WoodDeckSF: float = Field(..., ge=0, json_schema_extra={"example": 0.0})
    OpenPorchSF: float = Field(..., ge=0, json_schema_extra={"example": 61.0})
    EnclosedPorch: float = Field(..., ge=0, json_schema_extra={"example": 0.0})
    Neighborhood: str = Field(..., json_schema_extra={"example": "CollgCr"})
    BldgType: str = Field(..., json_schema_extra={"example": "1Fam"})
    HouseStyle: str = Field(..., json_schema_extra={"example": "2Story"})
    ExterQual: str = Field(..., json_schema_extra={"example": "Gd"})
    CentralAir: str = Field(..., json_schema_extra={"example": "Y"})
    KitchenQual: str = Field(..., json_schema_extra={"example": "Gd"})

    model_config = ConfigDict(populate_by_name=True)

# Membuat Skema Pydantic untuk Struktur Balasan (response payload) Prediksi API
class RumahResponse(BaseModel):
    harga_prediksi_usd: float
    model_version: str
    sumber_model: str
    latency_ms: float


# Middleware HTTP untuk Menyuntikkan Request ID Unik dan Mengukur Latency Total
@app.middleware("http")
async def catat_setiap_request(request: Request, call_next):
    mulai = time.perf_counter()
    request.state.request_id = uuid.uuid4().hex[:8]
    jawaban = await call_next(request)
    lama_ms = (time.perf_counter() - mulai) * 1000
    jawaban.headers["X-Request-ID"] = request.state.request_id
    if request.url.path not in ("/metrics", "/health"):
        log.info("http", extra={
            "request_id": request.state.request_id,
            "path": request.url.path,
            "status": jawaban.status_code,
            "latency_ms": round(lama_ms, 2),
        })
    return jawaban


# Membuat Function Pemeriksaan Kesehatan API
@app.get("/health")
def health():
    siap = kondisi["model"] is not None
    return {
        "status": "sehat" if siap else "belum siap",
        "model_version": kondisi["versi"],
        "sumber_model": kondisi["sumber"],
    }

# Membuat Function Melakukan Prediksi Harga Rumah
@app.post("/predict", response_model=RumahResponse)
def predict(permintaan: RumahRequest, request: Request):
    if kondisi["model"] is None:
        raise HTTPException(status_code=500, detail="Model belum dimuat di server.")

    mulai = time.perf_counter()
    METRIK_REQUEST.inc()

    try:
        # Konversi Pydantic ke Dict Menggunakan Nama Alias Asli ("1stFlrSF", "2ndFlrSF")
        baris = permintaan.model_dump(by_alias=True)
        
        tabel = pd.DataFrame([baris])
        
        # Pastikan Seluruh Feature yang Dibutuhkan Model ada Di DataFrame (isi dengan np.nan jika kosong)
        for kolom in config.FITUR:
            if kolom not in tabel.columns:
                tabel[kolom] = np.nan

        # Pastikan Seluruh Feature Kategorikal Bertipe String Secara Konsisten
        for kolom in config.FITUR_KATEGORIKAL:
            if kolom in tabel.columns:
                tabel[kolom] = tabel[kolom].astype(str)

        tabel = tabel[config.FITUR]
        
        # Eksekusi Prediksi
        prediksi = kondisi["model"].predict(tabel)[0]
        harga = float(prediksi)

    except Exception as salah:
        METRIK_ERROR.inc()
        req_id = getattr(request.state, "request_id", "unknown")
        log.error("prediksi_gagal", extra={"request_id": req_id, "pesan": str(salah)})
        print(f"\n❌ DETAIL ERROR: {salah}\n")
        raise HTTPException(status_code=500, detail=f"Gagal melakukan prediksi: {str(salah)}")

    lama = time.perf_counter() - mulai
    METRIK_LATENSI.observe(lama)
    METRIK_HARGA.observe(harga)

    # Pantau Data Drift Berdasarkan Pergeseran Rata-rata GrLivArea 100 Request Terakhir
    kondisi["grlivarea_terakhir"].append(permintaan.GrLivArea)
    if len(kondisi["grlivarea_terakhir"]) > 100:
        kondisi["grlivarea_terakhir"].pop(0)
    
    rata2 = sum(kondisi["grlivarea_terakhir"]) / len(kondisi["grlivarea_terakhir"])
    METRIK_GRLIVAREA_RATA2.set(rata2)
    METRIK_DRIFT.set(abs(rata2 - GRLIVAREA_RATA2_TRAINING) / GRLIVAREA_RATA2_TRAINING)

    req_id = getattr(request.state, "request_id", "unknown")
    log.info("prediksi", extra={
        "request_id": req_id,
        "GrLivArea": permintaan.GrLivArea,
        "OverallQual": permintaan.OverallQual,
        "prediksi_usd": round(harga, 2),
        "model_version": kondisi["versi"],
        "latency_ms": round(lama * 1000, 2),
    })

    return RumahResponse(
        harga_prediksi_usd=round(harga, 2),
        model_version=kondisi["versi"],
        sumber_model=kondisi["sumber"],
        latency_ms=round(lama * 1000, 2),
    )

# Membuat Function Pemuatan Ulang Model dari Registry/File Secara Hotspot
@app.post("/admin/reload")
def reload_model():
    versi_lama = kondisi["versi"]
    muat_model()
    log.warning("model_dimuat_ulang", extra={
        "versi_lama": versi_lama, "versi_baru": kondisi["versi"]})
    return {
        "status": "model dimuat ulang",
        "versi_lama": versi_lama,
        "versi_sekarang": kondisi["versi"],
    }

# Membuat Function Metrik Kinerja Sistem Dalam Format Prometheus
@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)