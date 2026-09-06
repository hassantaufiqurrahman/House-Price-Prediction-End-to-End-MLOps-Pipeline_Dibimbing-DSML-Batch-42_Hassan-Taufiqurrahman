# Import Library
import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from . import config

# Satu Contoh Rumah untuk Dicoba
CONTOH_RUMAH = {
    "MSSubClass": "60",
    "LotArea": 8450,
    "OverallQual": 7,
    "OverallCond": 5,
    "YearBuilt": 2003,
    "YearRemodAdd": 2003,
    "TotalBsmtSF": 856,
    "1stFlrSF": 856,
    "2ndFlrSF": 854,
    "GrLivArea": 1710,
    "FullBath": 2,
    "HalfBath": 1,
    "BedroomAbvGr": 3,
    "TotRmsAbvGrd": 8,
    "Fireplaces": 0,
    "GarageCars": 2,
    "GarageArea": 548,
    "WoodDeckSF": 0,
    "OpenPorchSF": 61,
    "EnclosedPorch": 0,
    "Neighborhood": "CollgCr",
    "BldgType": "1Fam",
    "HouseStyle": "2Story",
    "ExterQual": "Gd",
    "CentralAir": "Y",
    "KitchenQual": "Gd",
}

# Membuat Function Memuat Champion
def muat_champion():
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = MlflowClient()

    versi = client.get_model_version_by_alias(config.NAMA_MODEL, config.ALIAS_PRODUKSI)
    alamat = "models:/" + config.NAMA_MODEL + "@" + config.ALIAS_PRODUKSI
    
    model = mlflow.sklearn.load_model(alamat)
    return model, versi.version

# Membuat Function Membersihkan dan Mengurutkan Data
def bersihkan_dan_urutkan_data(df: pd.DataFrame) -> pd.DataFrame:
    tabel = df.copy()
    
    # Pastikan Seluruh Feature Terdaftar Ada, Jika Tidak Ada maka Isi Dengan "np.nan"
    for kolom in config.FEATURE:
        if kolom not in tabel.columns:
            tabel[kolom] = np.nan

    # Pastikan Kolom Kategorikal Konsisten Bertipe String (abaikan NaN)
    for kolom in config.FEATURE_KATEGORIKAL:
        if kolom in tabel.columns:
            tabel[kolom] = tabel[kolom].apply(lambda x: str(x) if pd.notna(x) else x)

    # Urutkan Kolom Sesuai Urutan Di "config.FEATURE"
    return tabel[config.FEATURE]

# Membuat Function Main untuk Menjalankan Seluruh Proses
def main():
    model, versi = muat_champion()
    print("=" * 60)
    print("Model yang dimuat : @" + config.ALIAS_PRODUKSI + " -> versi " + str(versi))
    print("=" * 60)

    # Pengujian Dengan "CONTOH_RUMAH" (Dictionary)
    tabel = pd.DataFrame([CONTOH_RUMAH])
    tabel = bersihkan_dan_urutkan_data(tabel)
    harga = model.predict(tabel)[0]

    print("\n[1] Contoh Rumah Uji:")
    for kunci in list(CONTOH_RUMAH.keys())[:5]:  
        print("  ", kunci, "=", CONTOH_RUMAH[kunci])
    print("   ...")
    print("\nPrediksi harga    : $ {:,.2f}".format(harga))

    # Pengujian Opsional dengan File "test_split.csv"
    try:
        df_test = pd.read_csv(config.FILE_TEST).head(3)
        tabel_test = bersihkan_dan_urutkan_data(df_test)
        prediksi_batch = model.predict(tabel_test)

        print("\n[2] Pengujian 3 sampel dari " + config.FILE_TEST + ":")
        for idx, pred in enumerate(prediksi_batch):
            aktual = df_test[config.TARGET].iloc[idx] if config.TARGET in df_test.columns else None
            aktual_str = f" | Aktual: $ {aktual:,.2f}" if aktual is not None else ""
            print(f"  Rumah #{idx+1} -> Prediksi: $ {pred:,.2f}{aktual_str}")
    except Exception as e:
        print(f"\n⚠️ Pengujian batch dilewati/eror: {e}")

    print("\n✓ Skrip predict.py selesai dijalankan.")


if __name__ == "__main__":
    main()