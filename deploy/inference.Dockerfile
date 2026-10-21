FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt
COPY inference/__init__.py inference/app.py inference/
COPY models/ models/
ENV MODEL_NAME=xgb
USER 10001
EXPOSE 8000
CMD ["uvicorn", "inference.app:app", "--host", "0.0.0.0", "--port", "8000"]
