# docker/Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src
COPY logs/ ./logs
COPY data/ ./data

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
