# Import Library
import argparse

import mlflow
from mlflow import MlflowClient

from . import config
from .train import pindah_champion


# Menampilkan Semua Versi Model Beserta MAE dan Aliasnya
def tampilkan_semua_versi(client):
    alias_per_versi = {}
    model_terdaftar = client.get_registered_model(config.NAMA_MODEL)
    
    if hasattr(model_terdaftar, "aliases") and model_terdaftar.aliases:
        for nama_alias, versi_target in model_terdaftar.aliases.items():
            nomor = int(versi_target)
            if nomor not in alias_per_versi:
                alias_per_versi[nomor] = []
            alias_per_versi[nomor].append("@" + nama_alias)

    daftar = client.search_model_versions("name='" + config.NAMA_MODEL + "'")
    baris = []
    for v in daftar:
        run = client.get_run(v.run_id)
        mae = run.data.metrics.get("mae", 0)
        nomor = int(v.version)
        baris.append((nomor, run.info.run_name, mae, alias_per_versi.get(nomor, [])))
    baris.sort()

    print("\n  versi  run                 mae                alias")
    print("  " + "-" * 66)
    for versi, nama_run, mae, alias in baris:
        tanda = ""
        if "@" + config.ALIAS_PRODUKSI in alias:
            tanda = "  <-- dipakai API sekarang"
        print("  v{:<5d} {:<19s} $ {:>13,.2f}   {}{}".format(
            versi, str(nama_run)[:19], mae, " ".join(alias), tanda))
    return baris


def main():
    pilihan = argparse.ArgumentParser(description="Rollback model champion")
    pilihan.add_argument("--ke-versi", type=int, default=None)
    argumen = pilihan.parse_args()

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = MlflowClient()

    print("=" * 62)
    print("ROLLBACK — memindahkan alias @" + config.ALIAS_PRODUKSI)
    print("=" * 62)

    tampilkan_semua_versi(client)

    champion = client.get_model_version_by_alias(config.NAMA_MODEL, config.ALIAS_PRODUKSI)
    versi_sekarang = int(champion.version)

    tujuan = argumen.ke_versi
    if tujuan is None:
        try:
            sebelumnya = client.get_model_version_by_alias(
                config.NAMA_MODEL, config.ALIAS_SEBELUMNYA)
            tujuan = int(sebelumnya.version)
            print("\n  @" + config.ALIAS_SEBELUMNYA + " menunjuk ke v" + str(tujuan))
        except Exception:
            raise SystemExit(
                "❌ Belum ada catatan champion sebelumnya.\n"
                "   Sebutkan tujuannya langsung, misal: python rollback.py --ke-versi 3"
            )

    if tujuan == versi_sekarang:
        raise SystemExit("❌ v" + str(tujuan) + " sudah jadi champion sekarang.")

    print("\n  ROLLBACK: v" + str(versi_sekarang) + "  ->  v" + str(tujuan))
    pindah_champion(client, tujuan)

    print("\n  ⚠ API masih memegang model lama di memori (dimuat sekali saat startup).")
    print("    Jalankan ini supaya API ikut berubah:")
    print("      curl -X POST http://localhost:8000/admin/reload")


if __name__ == "__main__":
    main()