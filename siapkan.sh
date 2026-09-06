#!/usr/bin/env bash
# siapkan.sh — menyalin repo ini ke path bersih lalu menyiapkan conda env.
# Dijalankan SEKALI. Setelah ini, semua pekerjaan dilakukan di ~/Day49_MLOps_Demo
#
#   bash siapkan.sh
#
# Kenapa perlu? Karena nama folder materi mengandung tanda titik dua (:),
# dan Docker memakai ':' sebagai pemisah source:target.

set -e
TUJUAN="$HOME/Day49_MLOps_Demo"
ASAL="$(cd "$(dirname "$0")" && pwd)"
ENV_NAMA="day49-mlops"

if ! command -v conda >/dev/null 2>&1; then
  echo "✗ conda tidak ditemukan. Pasang Miniconda dulu:"
  echo "    https://www.anaconda.com/download/success"
  echo "  lalu buka terminal BARU dan jalankan ulang: bash siapkan.sh"
  exit 1
fi

echo "menyalin  : $ASAL"
echo "        -> $TUJUAN"
mkdir -p "$TUJUAN"
rsync -a --exclude .venv --exclude __pycache__ "$ASAL"/ "$TUJUAN"/ 2>/dev/null \
  || cp -R "$ASAL"/. "$TUJUAN"/

cd "$TUJUAN"

if ! conda env list | grep -qE "^${ENV_NAMA}[[:space:]]"; then
  echo "membuat conda env '${ENV_NAMA}' (unduhan ±200 MB)..."
  conda create -n "$ENV_NAMA" python=3.12 -y
  conda run -n "$ENV_NAMA" pip install --quiet --upgrade pip
  conda run -n "$ENV_NAMA" pip install -r requirements.txt
fi

conda run -n "$ENV_NAMA" python check_setup.py

echo
echo "✓ Siap. Mulai bekerja dari sana:"
echo "    cd ~/Day49_MLOps_Demo && conda activate ${ENV_NAMA}"
