FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install NumPy FIRST, then TensorFlow
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir numpy==1.23.5 && \
    pip install --no-cache-dir -r requirements.txt

# Verify installations
RUN python -c "import numpy; print(f'NumPy: {numpy.__version__}'); import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"

COPY api/ ./api/
COPY models/ ./models/

WORKDIR /app/api

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "600"]