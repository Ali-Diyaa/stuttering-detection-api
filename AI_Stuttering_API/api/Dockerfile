# Use Python 3.10 for TensorFlow compatibility
FROM python:3.10-slim

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create models directory and ensure models are copied
RUN mkdir -p /app/models

# Expose port
EXPOSE 8000

# Run with better settings for production
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]