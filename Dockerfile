FROM python:3.12-slim

WORKDIR /app

# Dependency dulu (jarang berubah) -> layer ini di-cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Baru kode aplikasi (sering berubah)
COPY src/ ./src
COPY api/ ./api

# Model hasil ekspor dari `python -m src.train`
COPY models/ ./models

# Folder untuk log (di-mount dari luar lewat docker-compose)
RUN mkdir -p logs

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
