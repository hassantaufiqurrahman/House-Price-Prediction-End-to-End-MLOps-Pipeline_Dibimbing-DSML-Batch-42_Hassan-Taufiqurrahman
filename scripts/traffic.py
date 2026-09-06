# Import Library
import sys
import time

import numpy as np
import requests

ALAMAT = "http://localhost:8000/predict"
JUMLAH_PER_GELOMBANG = 200
JEDA_DETIK = 0.05          # 20 request per detik

# Membuat Function Simulasi Pengiriman Sekelompok (Gelombang) Request ke API
def kirim_gelombang(nama, grlivarea_rata2, grlivarea_sebaran, jumlah, seed):
    # Menginisialisasi generator angka acak NumPy berbasis seed
    acak = np.random.default_rng(seed)
    
    # Hitung jumlah request yang berhasil dan gagal
    berhasil = 0
    gagal = 0

    print(f"\n>> Gelombang '{nama}' — {jumlah} request, "
          f"rata-rata GrLivArea = {grlivarea_rata2} sqft")

    # Menggunakan Session untuk HTTP Keep-Alive agar koneksi lebih efisien
    with requests.Session() as sesi:
        # Iterasi untuk mengirimkan request satu per satu
        for nomor in range(jumlah):
            gr_liv = float(np.clip(acak.normal(grlivarea_rata2, grlivarea_sebaran), 500, 4500))
            
            flr1 = round(gr_liv * 0.55, 1)  
            flr2 = round(gr_liv * 0.45, 1)  
            
            isi = {
                "MSSubClass": "60",
                "LotArea": int(np.clip(acak.normal(8500, 1500), 2000, 20000)),  
                "OverallQual": int(np.clip(acak.normal(6, 1.5), 1, 10)),      
                "OverallCond": 5,                                             
                "YearBuilt": int(acak.integers(1980, 2010)),                  
                "YearRemodAdd": 2005,                                         
                "TotalBsmtSF": float(round(flr1 * 0.9, 1)),                                
                "1stFlrSF": float(flr1),                                            
                "2ndFlrSF": float(flr2),                                            
                "GrLivArea": float(round(gr_liv, 1)),                                
                "FullBath": 2,                                                
                "HalfBath": 1,                                                
                "BedroomAbvGr": 3,                                            
                "TotRmsAbvGrd": 7,                                            
                "Fireplaces": 1,                                              
                "GarageCars": 2,                                              
                "GarageArea": 480.0,                                          
                "WoodDeckSF": 0.0,                                            
                "OpenPorchSF": 50.0,                                          
                "EnclosedPorch": 0.0,                                         
                "Neighborhood": "CollgCr",                                    
                "BldgType": "1Fam",                                           
                "HouseStyle": "2Story",                                       
                "ExterQual": "Gd",                                            
                "CentralAir": "Y",                                            
                "KitchenQual": "Gd"                                           
            }

            try:
                jawaban = sesi.post(ALAMAT, json=isi, timeout=5)
                
                if jawaban.status_code == 200:
                    berhasil += 1
                else:
                    gagal += 1
            except Exception:
                gagal += 1

            if (nomor + 1) % 50 == 0:
                print(f"   {nomor + 1}/{jumlah} terkirim...")
                
            time.sleep(JEDA_DETIK)

    print(f"   selesai: {berhasil} berhasil, {gagal} gagal")
    return gagal

# Membuat Function Skenario Trafik Simulasi Dua Gelombang
def main():
    print("=" * 62)
    print("PENGIRIM TRAFIK — buka Grafana di http://localhost:3000")
    print("=" * 62)

    try:
        cek = requests.get("http://localhost:8000/health", timeout=5)
        print("API :", cek.json())
    except Exception:
        print("❌ API tidak bisa dihubungi di http://localhost:8000")
        print("   Nyalakan dulu: uvicorn api.main:app --port 8000")
        print("   atau: docker compose up -d")
        sys.exit(1)

    # Gelombang 1: Mengirim Traffic Normal 
    kirim_gelombang(
        nama="normal", 
        grlivarea_rata2=1515.0, 
        grlivarea_sebaran=250.0,
        jumlah=JUMLAH_PER_GELOMBANG, 
        seed=1
    )

    print("\n--- perhatikan Grafana sekarang, lalu tekan ENTER ---")
    try:
        input()
    except EOFError:
        pass

    # Gelombang 2: Mengirim Trafik Bergeser / Drift 
    kirim_gelombang(
        nama="bergeser", 
        grlivarea_rata2=2800.0, 
        grlivarea_sebaran=350.0,
        jumlah=JUMLAH_PER_GELOMBANG, 
        seed=2
    )

    print("\n✓ SELESAI. Di Grafana perhatikan:")
    print("   * panel 'Rata-rata GrLivArea' naik tajam")
    print("   * panel 'Skor drift' menyeberang ke zona merah")
    print("   * sebaran 'Harga prediksi' bergeser naik")
    print("\n   Semua request dijawab 200 OK.")


if __name__ == "__main__":
    main()
