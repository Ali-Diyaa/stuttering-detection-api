FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Debug: Check what was copied
RUN echo "=== Checking directory structure ===" && \
    echo "Contents of /app:" && ls -la /app && \
    echo "Contents of /app/AI_Stuttering_API:" && ls -la /app/AI_Stuttering_API && \
    echo "Contents of /app/AI_Stuttering_API/models:" && ls -la /app/AI_Stuttering_API/models && \
    echo "Contents of /app/AI_Stuttering_API/api:" && ls -la /app/AI_Stuttering_API/api

WORKDIR /app/AI_Stuttering_API/api

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]