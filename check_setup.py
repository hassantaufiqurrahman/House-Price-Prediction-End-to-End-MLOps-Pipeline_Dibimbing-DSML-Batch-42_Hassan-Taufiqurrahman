import importlib
import os
import platform
import shutil
import subprocess
import sys

PAKET = [
    "mlflow", "sklearn", "pandas", "numpy", 
    "matplotlib", "seaborn", "scipy", "joblib",
    "fastapi", "uvicorn", "pydantic", "prometheus_client", 
    "requests", "anyio"
]

def cek(nama_hal, berhasil, keterangan=""):
    tanda = "✓" if berhasil else "✗"
    print("  " + tanda + "  " + nama_hal.ljust(28) + keterangan)
    return berhasil


def main():
    print("=" * 58)
    print("CEK PERSIAPAN — Day 49 : Python-Powered MLOps")
    print("=" * 58)
    semua_ok = True

    # 1. Python
    versi = sys.version_info
    ok = versi.major == 3 and versi.minor >= 10
    semua_ok = cek("Python >= 3.10", ok, platform.python_version()) and semua_ok

    # 2. environment terisolasi — conda (utama) atau venv (masih diterima)
    nama_conda = os.environ.get("CONDA_DEFAULT_ENV", "")
    di_venv = sys.prefix != sys.base_prefix
    aktif = bool(nama_conda) or di_venv
    ket = ("conda: " + nama_conda) if nama_conda else (
        sys.prefix if di_venv
        else "belum aktif — jalankan: conda activate day49-mlops")
    semua_ok = cek("environment aktif", aktif, ket) and semua_ok

    # 3. paket
    print("\n  Paket Python:")
    for nama in PAKET:
        try:
            modul = importlib.import_module(nama)
            v = getattr(modul, "__version__", "ok")
            cek(nama, True, str(v))
        except Exception:
            semua_ok = cek(nama, False, "belum terpasang") and semua_ok

    # 4. docker
    print("\n  Docker:")
    ada_docker = shutil.which("docker") is not None
    if ada_docker:
        try:
            hasil = subprocess.run(["docker", "ps"], capture_output=True,
                                   text=True, timeout=15)
            jalan = hasil.returncode == 0
            semua_ok = cek("Docker Desktop menyala", jalan,
                           "" if jalan else "buka aplikasi Docker Desktop dulu") and semua_ok
        except Exception:
            semua_ok = cek("Docker Desktop menyala", False, "tidak merespons") and semua_ok
    else:
        semua_ok = cek("perintah docker ada", False, "Docker Desktop belum terpasang") and semua_ok

    print("\n" + "=" * 58)
    if semua_ok:
        print("✓ SEMUA SIAP. Sampai jumpa di kelas!")
    else:
        print("✗ MASIH ADA YANG KURANG.")
        print("  Baca SETUP.md bagian 'Kalau ada yang gagal'.")
        print("  Docker yang gagal tidak menghalangi Part 1-5 — kabari mentor saja.")
    print("=" * 58)


if __name__ == "__main__":
    main()
