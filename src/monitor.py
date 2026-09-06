# Import Library
import os

import mlflow
import pandas as pd
from mlflow import MlflowClient
from scipy.stats import ks_2samp
from sklearn.metrics import mean_absolute_error

from . import config


# Mengambil Feature dari Log Request. Jika log belum cukup maka Menggunakan File "live_drift.csv"
def ambil_input_live() -> pd.DataFrame:
    if os.path.exists(config.FILE_LOG):
        catatan = pd.read_json(config.FILE_LOG, lines=True)
        if "event" in catatan.columns:
            catatan = catatan[catatan["event"] == "prediksi"]
        cukup = True
        for kolom in config.FITUR:
            if kolom not in catatan.columns:
                cukup = False
        if cukup and len(catatan) >= 30:
            total = len(catatan)
            catatan = catatan.tail(config.JENDELA_MONITOR)
            print(
                "  sumber input live : "
                + config.FILE_LOG
                + " ("
                + str(len(catatan))
                + " request terakhir dari "
                + str(total)
                + ")"
            )
            return catatan[config.FITUR]

    print(
        "  sumber input live : "
        + config.FILE_LIVE_DRIFT
        + "  (log belum cukup, pakai file contoh)"
    )
    return pd.read_csv(config.FILE_LIVE_DRIFT)[config.FITUR]

# Membandingkan Dua Distribusi Feature Numerik Menggunakan KS-Test
def cek_drift(train: pd.DataFrame, live: pd.DataFrame) -> bool:
    print("\n[1] CEK DRIFT — apakah data yang masuk masih mirip data training?")
    print("\n  kolom         mean train        mean live      geser     p-value   status")
    print("  " + "-" * 74)

    ada_drift = False
    for kolom in config.FITUR_NUMERIK:
        if kolom not in train.columns or kolom not in live.columns:
            continue

        mean_train = train[kolom].mean()
        mean_live = live[kolom].mean()
        if mean_train == 0:
            geser = 0.0
        else:
            geser = (mean_live - mean_train) / mean_train * 100

        hasil = ks_2samp(train[kolom].dropna(), live[kolom].dropna())
        if hasil.pvalue < config.AMBANG_PVALUE:
            status = "DRIFT"
            ada_drift = True
        else:
            status = "aman"

        print(
            "  {:<12s} {:>12.2f} {:>15.2f}  {:>+8.1f}%  {:>10.2g}   {}".format(
                kolom, mean_train, mean_live, geser, hasil.pvalue, status
            )
        )

    if ada_drift:
        print("\n  ⚠ Ada kolom yang distribusinya sudah bergeser.")
        print(
            "    Model tidak error — dia cuma sedang ditanya soal yang belum pernah dia pelajari."
        )
    else:
        print("\n  ✓ Semua kolom numerik masih mirip data training.")
    return ada_drift

# Menghitung MAE model Champion Pada Data Live Berlabel (jika tersedia)
def cek_mae(train_mae_champion: float):
    print("\n[2] CEK MAE — kalau harga sebenarnya sudah kita ketahui")

    if not os.path.exists(config.FILE_LIVE_DRIFT):
        print("  (file berlabel tidak ada, dilewati)")
        return

    live = pd.read_csv(config.FILE_LIVE_DRIFT)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = MlflowClient()
    versi = client.get_model_version_by_alias(config.NAMA_MODEL, config.ALIAS_PRODUKSI)
    model = mlflow.pyfunc.load_model(
        "models:/" + config.NAMA_MODEL + "@" + config.ALIAS_PRODUKSI
    )

    for kolom in config.FITUR_KATEGORIKAL:
        if kolom in live.columns:
            live[kolom] = live[kolom].astype(str)

    data_prediksi = live[config.FITUR].copy()

    kolom_int = [
        "LotArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
        "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath",
        "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr",
        "TotRmsAbvGrd", "Fireplaces", "GarageCars", "GarageArea", "WoodDeckSF",
        "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea",
        "MiscVal", "MoSold", "YrSold", "YearBuilt", "YearRemodAdd",
        "OverallQual", "OverallCond"
    ]
    for col in kolom_int:
        if col in data_prediksi.columns:
            data_prediksi[col] = data_prediksi[col].fillna(0).round().astype("int64")

    kolom_float = ["LotFrontage", "MasVnrArea", "GarageYrBlt"]
    for col in kolom_float:
        if col in data_prediksi.columns:
            data_prediksi[col] = data_prediksi[col].astype("float64")

    prediksi = model.predict(data_prediksi)
    mae_live = mean_absolute_error(live[config.TARGET], prediksi)

    print("\n  model champion    : v" + str(versi.version))
    print("  MAE saat training : $ {:>15,.2f}".format(train_mae_champion))
    print("  MAE di data live  : $ {:>15,.2f}".format(mae_live))
    if train_mae_champion > 0:
        kali = mae_live / train_mae_champion
        print("  memburuk          : {:.1f}x".format(kali))
        if kali > 1.5:
            print("\n  ❌ Model sudah tidak bisa dipercaya untuk data sekarang.")
            print(
                "     Perhatikan: tidak ada satu pun error atau exception yang muncul."
            )
            print("     API tetap menjawab HTTP 200 — dengan angka yang salah.")

# Membaca MAE Model Champion dari MLflow Tracking Run
def ambil_mae_champion() -> float:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = MlflowClient()
    versi = client.get_model_version_by_alias(config.NAMA_MODEL, config.ALIAS_PRODUKSI)
    run = client.get_run(versi.run_id)
    return float(run.data.metrics.get("mae", 0))


def main():
    print("=" * 74)
    print("MONITORING — apakah model masih waras?")
    print("=" * 74)

    train = pd.read_csv(config.FILE_TRAIN)
    live = ambil_input_live()
    print("  jumlah baris train:", len(train), "| live:", len(live))

    cek_drift(train, live)
    cek_mae(ambil_mae_champion())

    print("\n" + "=" * 74)
    print("Kesimpulan: lapisan DATA (drift) memberi peringatan lebih dulu")
    print("daripada lapisan MODEL (MAE) — karena MAE butuh harga sebenarnya,")
    print("dan harga sebenarnya baru diketahui setelah transaksi terjadi.")


if __name__ == "__main__":
    main()