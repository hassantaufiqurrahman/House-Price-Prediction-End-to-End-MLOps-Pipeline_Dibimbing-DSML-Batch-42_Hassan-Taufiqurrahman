# Import Library
import argparse
import hashlib
import os
import shutil
import subprocess

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config

# Membuat Function untuk Mengambil Git SHA
def ambil_git_sha():
    try:
        hasil = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if hasil.returncode == 0:
            return hasil.stdout.strip()
    except Exception:
        pass
    return "belum-pakai-git"

# Membuat Function untuk Mengambil Hash Data
def ambil_hash_data(path_file):
    isi = open(path_file, "rb").read()
    return hashlib.md5(isi).hexdigest()[:12]

# Membuat Function untuk Membangun Model
def buat_model(nama_model, jumlah_pohon, kedalaman, alpha):
    # Melakukan Preprocessor untuk Feature Numerik
    pipeline_numerik = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Melakukan Preprocessor untuk Feature Kategorikal
    pipeline_kategorikal = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # Menggabungkan Preprocessor
    preprocessor = ColumnTransformer([
        ("num", pipeline_numerik, config.FITUR_NUMERIK),
        ("cat", pipeline_kategorikal, config.FITUR_KATEGORIKAL),
    ])

    # Pemilihan Model Machine Learning
    if nama_model == "linear":
        inti = LinearRegression()
    elif nama_model == "ridge":
        inti = Ridge(alpha=alpha, random_state=config.RANDOM_STATE)
    elif nama_model == "lasso":
        inti = Lasso(alpha=alpha, random_state=config.RANDOM_STATE)
    elif nama_model == "rf":
        inti = RandomForestRegressor(
            n_estimators=jumlah_pohon,
            max_depth=kedalaman,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        raise SystemExit("❌ --model hanya boleh: 'linear', 'ridge', 'lasso', atau 'rf'")

    # Menyatukan Preprocessor dan Model ke dalam Pipeline Utama
    langkah = [
        ("preprocessor", preprocessor),
        ("model", inti),
    ]
    return Pipeline(langkah)

# Membuat Function untuk Memindahkan Alias Champion
def pindah_champion(client, versi_tujuan):
    lama = None
    try:
        lama = client.get_model_version_by_alias(config.NAMA_MODEL, config.ALIAS_PRODUKSI)
    except Exception:
        lama = None

    if lama is not None and int(lama.version) != int(versi_tujuan):
        client.set_registered_model_alias(
            config.NAMA_MODEL, config.ALIAS_SEBELUMNYA, str(lama.version))
        print("      @" + config.ALIAS_SEBELUMNYA + " -> v" + str(lama.version))

    client.set_registered_model_alias(
        config.NAMA_MODEL, config.ALIAS_PRODUKSI, str(versi_tujuan))
    print("      @" + config.ALIAS_PRODUKSI + " -> v" + str(versi_tujuan))
    ekspor_champion(client)

# Membuat Function untuk Mengekspor Model Champion
def ekspor_champion(client):
    versi = client.get_model_version_by_alias(config.NAMA_MODEL, config.ALIAS_PRODUKSI)
    model = mlflow.sklearn.load_model(
        "models:/" + config.NAMA_MODEL + "@" + config.ALIAS_PRODUKSI
    )
    if os.path.isdir(config.FOLDER_MODEL_EKSPOR):
        shutil.rmtree(config.FOLDER_MODEL_EKSPOR)
    os.makedirs(config.FOLDER_MODEL_EKSPOR, exist_ok=True)
    joblib.dump(model, os.path.join(config.FOLDER_MODEL_EKSPOR, "model.joblib"))
    
    with open(os.path.join(config.FOLDER_MODEL_EKSPOR, "VERSION"), "w") as tulis:
        tulis.write(str(versi.version))
        
    print("  [ekspor] champion v" + str(versi.version) + " -> " + config.FOLDER_MODEL_EKSPOR)

# Membuat Function Main untuk Menjalankan Seluruh Proses
def main():
    pilihan = argparse.ArgumentParser(description="Latih model harga rumah Kaggle")
    pilihan.add_argument("--model", default="rf", help="linear, ridge, lasso, atau rf")
    pilihan.add_argument("--alpha", type=float, default=1.0, help="Hiperparameter L2/L1 penalty untuk Ridge/Lasso")
    pilihan.add_argument("--n-estimators", type=int, default=config.N_ESTIMATORS)
    pilihan.add_argument("--max-depth", type=int, default=config.MAX_DEPTH)
    pilihan.add_argument("--data", default=config.FILE_TRAIN)
    pilihan.add_argument("--nama-run", default=None)
    pilihan.add_argument("--promote", action="store_true",
                         help="jadikan model ini @champion (walaupun lebih buruk)")
    argumen = pilihan.parse_args()

    print("=" * 62)
    print("POS 4-6 : latih -> ukur -> gerbang mutu -> daftarkan")
    print("=" * 62)

    # Memuat Data Train & Test
    train = pd.read_csv(argumen.data)
    test = pd.read_csv(config.FILE_TEST)

    # Memastikan Seluruh Kolom Kategorikal Konsisten Bertipe String
    for kolom in config.FITUR_KATEGORIKAL:
        if kolom in train.columns:
            train[kolom] = train[kolom].astype(str)
        if kolom in test.columns:
            test[kolom] = test[kolom].astype(str)

    print("\n[1] Data")
    print("  train :", len(train), "baris dari", argumen.data)
    print("  test  :", len(test), "baris dari", config.FILE_TEST)

    X_train = train[config.FITUR]
    y_train = train[config.TARGET]
    X_test = test[config.FITUR]
    y_test = test[config.TARGET]

    # Mengatur MLflow Tracking & Experiment
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.NAMA_EXPERIMENT)
    client = MlflowClient()

    nama_run = argumen.nama_run
    if nama_run is None:
        if argumen.model in ["ridge", "lasso"]:
            nama_run = argumen.model + "-alpha-" + str(argumen.alpha)
        elif argumen.model == "rf":
            nama_run = argumen.model + "-" + str(argumen.n_estimators)
        else:
            nama_run = argumen.model

    with mlflow.start_run(run_name=nama_run) as run:
        print("\n[2] MLflow run :", nama_run, "(" + run.info.run_id[:8] + ")")

        # Melatih Model
        pipeline = buat_model(argumen.model, argumen.n_estimators, argumen.max_depth, argumen.alpha)
        pipeline.fit(X_train, y_train)
        print("[3] Model dilatih.")

        # Melakukan Evaluasi Model
        prediksi = pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, prediksi)
        rmse = mean_squared_error(y_test, prediksi) ** 0.5
        r2 = r2_score(y_test, prediksi)

        print("[4] Hasil di data test:")
        print("      MAE  : $ {:,.2f}".format(mae))
        print("      RMSE : $ {:,.2f}".format(rmse))
        print("      R2   : {:.4f}".format(r2))

        # Menyimpan Parameter, Metriks Hasil Evaluasi Model, dan Tags ke MLflow
        mlflow.log_param("model", argumen.model)
        if argumen.model in ["ridge", "lasso"]:
            mlflow.log_param("alpha", argumen.alpha)
        elif argumen.model == "rf":
            mlflow.log_param("n_estimators", argumen.n_estimators)
            mlflow.log_param("max_depth", argumen.max_depth)
            
        mlflow.log_param("random_state", config.RANDOM_STATE)
        mlflow.log_param("file_data", argumen.data)

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        mlflow.set_tag("git_sha", ambil_git_sha())
        mlflow.set_tag("data_md5", ambil_hash_data(argumen.data))
        print("[5] Params, metrics, dan tags tercatat.")

        # Melakukan Quality Gate
        print("\n[6] 🚧 Gerbang mutu: MAE harus <= $ {:,.2f}".format(config.AMBANG_MAE))
        if mae > config.AMBANG_MAE:
            mlflow.set_tag("lolos_gate", "tidak")
            print("      ❌ TIDAK LOLOS. Model tidak didaftarkan ke registry.")
            raise SystemExit(
                "GATE GAGAL: MAE $ {:,.2f} > ambang $ {:,.2f}".format(
                    mae, config.AMBANG_MAE)
            )
        mlflow.set_tag("lolos_gate", "ya")
        print("      ✓ LOLOS. Model boleh masuk registry.")

        # Mendaftarkan Model ke MLflow Registry
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name=config.NAMA_MODEL,
            input_example=X_train.head(2),
            skops_trusted_types=["numpy.dtype"],
        )
        print("\n[7] Model didaftarkan ke registry:", config.NAMA_MODEL)

    # Memeriksa Versi Terbaru Model di Registry
    semua_versi = client.search_model_versions("name='" + config.NAMA_MODEL + "'")
    versi_baru = 0
    for v in semua_versi:
        if int(v.version) > versi_baru:
            versi_baru = int(v.version)
    print("      versi baru :", versi_baru)

    champion_sekarang = None
    try:
        champion_sekarang = client.get_model_version_by_alias(
            config.NAMA_MODEL, config.ALIAS_PRODUKSI)
    except Exception:
        champion_sekarang = None

    if champion_sekarang is None:
        print("\n[8] Belum ada champion -> model pertama otomatis jadi champion.")
        pindah_champion(client, versi_baru)
    elif argumen.promote:
        print("\n[8] --promote dipakai -> champion dipindah ke v" + str(versi_baru))
        pindah_champion(client, versi_baru)
    else:
        print("\n[8] Champion TIDAK diubah (masih v" + str(champion_sekarang.version) + ").")
        print("      Model ini jadi kandidat @challenger.")
        client.set_registered_model_alias(
            config.NAMA_MODEL, config.ALIAS_KANDIDAT, str(versi_baru))

    print("\n✓ SELESAI. Buka UI-nya:  python -m mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()