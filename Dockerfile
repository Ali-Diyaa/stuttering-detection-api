FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy everything
COPY . .

# Install requirements (requirements.txt is at api/requirements.txt)
RUN pip install --no-cache-dir -r api/requirements.txt

EXPOSE 8000

# Run from api folder
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]