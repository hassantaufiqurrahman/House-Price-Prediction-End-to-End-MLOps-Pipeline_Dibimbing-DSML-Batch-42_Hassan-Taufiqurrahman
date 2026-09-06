# Import Library
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from . import config

# Membuat Function untuk Memuat Data
def muat_data():
    if not os.path.exists(config.FILE_RAW):
        raise SystemExit(
            f"❌ FILE TIDAK DITEMUKAN: File '{config.FILE_RAW}' belum ada.\n"
            "   Silakan unduh 'train.csv' dari Kaggle, rename menjadi 'house_prices_dataset', dan simpan di folder 'data/'."
        )

    print(f"  [muat] membaca data dari {config.FILE_RAW}...")
    tabel = pd.read_csv(config.FILE_RAW)

    # Mengubah Tipe Data Kolom Kategorikal Menjadi String secara Eksplisit
    for kolom in config.FITUR_KATEGORIKAL:
        if kolom in tabel.columns:
            tabel[kolom] = tabel[kolom].astype(str)

    return tabel


# Membuat Function untuk Memeriksa Data
def periksa_data(tabel):
    print("  [periksa] jumlah baris :", len(tabel))
    if len(tabel) == 0:
        raise SystemExit("❌ VALIDASI GAGAL: tabelnya kosong.")

    # Memastikan Seluruh Kolom Feature & Target ada di Dataset
    kolom_wajib = list(config.FITUR) + [config.TARGET]
    for kolom in kolom_wajib:
        if kolom not in tabel.columns:
            raise SystemExit(
                f"❌ VALIDASI GAGAL: kolom '{kolom}' tidak ada di dataset."
            )

    # Memastikan Nilai Target Valid
    jumlah_target_kosong = int(tabel[config.TARGET].isna().sum())
    jumlah_target_aneh = int((tabel[config.TARGET] <= 0).sum())

    print("  [periksa] target kosong :", jumlah_target_kosong)
    print("  [periksa] target <= 0   :", jumlah_target_aneh)

    if jumlah_target_kosong > 0 or jumlah_target_aneh > 0:
        raise SystemExit("❌ VALIDASI GAGAL: Nilai SalePrice tidak valid/kosong.")

    jumlah_fitur_kosong = int(tabel[config.FITUR].isna().sum().sum())
    print(
        "  [periksa] total NaN pada fitur :",
        jumlah_fitur_kosong,
        "(akan di-impute di pipeline)",
    )

    print("  [periksa] ✓ data lolos validasi")


# Membuat Function untuk Membagi Data dan Menyimpan ke CSV
def bagi_dan_simpan(tabel):
    train, test = train_test_split(
        tabel,
        test_size=config.UKURAN_TEST,
        random_state=config.RANDOM_STATE,
    )

    # Memastikan Tidak Ada Baris yang Hilang Saat Split Dataset
    if len(train) + len(test) != len(tabel):
        raise SystemExit("❌ ada baris yang hilang saat split dataset!")

    # Membuat Data Live Drift dari Sebagian Data Test (Simulasi Drift)
    live_drift = test.copy()
    fitur_geser = ['GrLivArea', 'LotArea', 'TotalBsmtSF']
    for col in fitur_geser:
        if col in live_drift.columns:
            live_drift[col] = np.round(live_drift[col] * 1.35, 2)

    # Simpan File ke Directory
    os.makedirs(config.FOLDER_DATA, exist_ok=True)
    train.to_csv(config.FILE_TRAIN, index=False)
    test.to_csv(config.FILE_TEST, index=False)
    live_drift.to_csv(config.FILE_LIVE_DRIFT, index=False)

    print("  [simpan] train      :", len(train), "baris ->", config.FILE_TRAIN)
    print("  [simpan] test       :", len(test), "baris ->", config.FILE_TEST)
    print("  [simpan] live_drift :", len(live_drift), "baris ->", config.FILE_LIVE_DRIFT)


# Membuat Function Main untuk Menjalankan Seluruh Proses
def main():
    print("=" * 62)
    print("POS 1-3 : muat data Kaggle -> periksa -> bagi -> buat drift")
    print("=" * 62)

    print("\n[1] Memuat data Kaggle House Prices...")
    tabel = muat_data()

    # Cetak Preview Aman
    kolom_preview = [c for c in ["Id"] + config.FITUR_NUMERIK[:3] + [config.TARGET] if c in tabel.columns]
    print(tabel[kolom_preview].head().to_string(index=False))

    print("\n[2] Memeriksa data...")
    periksa_data(tabel)

    print("\n[3] Membagi & menyimpan data split + live_drift...")
    bagi_dan_simpan(tabel)

    print("\n✓ SELESAI. Lanjut ke Part 2: python -m src.train")


if __name__ == "__main__":
    main()