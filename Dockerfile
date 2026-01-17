FROM python:3.10-slim

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements from root of repo
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the AI_Stuttering_API folder (which contains api/ and models/)
COPY AI_Stuttering_API/ ./AI_Stuttering_API/

# Set working directory to the api folder inside AI_Stuttering_API
WORKDIR /app/AI_Stuttering_API/api

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]