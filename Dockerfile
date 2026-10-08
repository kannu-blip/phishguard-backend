# PhishGuard AI backend. Train first (python train.py ...) so model.joblib exists
# in this folder; it is copied into the image. Without it /scan returns 503.
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Most hosts (Render, Fly, Cloud Run...) inject $PORT.
EXPOSE 8000
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]
