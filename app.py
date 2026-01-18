import io
import librosa
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Audio Feature Extractor API",
    description="Extracts 39 MFCC features from audio files - Shape (1, 39, 1)",
    version="1.0.0"
)

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_features_from_audio(audio, sr):
    """YOUR EXACT FUNCTION - Returns shape (1, 39, 1)"""
    try:
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        mfcc_mean = np.mean(mfcc.T, axis=0)
        delta_mean = np.mean(delta.T, axis=0)
        delta2_mean = np.mean(delta2.T, axis=0)

        features = np.concatenate([mfcc_mean, delta_mean, delta2_mean])

        return features.reshape(1, -1, 1)  # Shape: (1, 39, 1)

    except Exception as e:
        raise Exception(f"Error extracting features: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint - API info"""
    return {
        "service": "Audio Feature Extractor API",
        "status": "running",
        "endpoint": "POST /extract-features/",
        "feature_shape": "(1, 39, 1)",
        "features_total": 39,
        "features_breakdown": "13 MFCCs + 13 Deltas + 13 Delta-Deltas"
    }

@app.get("/health")
async def health():
    """Health check for Koyeb monitoring"""
    return {
        "status": "healthy",
        "service": "Audio Feature Extractor",
        "version": "1.0.0"
    }

@app.post("/extract-features/")
async def extract_features(file: UploadFile = File(...)):
    """
    Extract 39 MFCC features from audio file.
    Returns features with shape (1, 39, 1)
    """
    try:
        print(f"Processing: {file.filename} ({file.content_type})")
        
        # Check file
        if not file.filename:
            return JSONResponse(
                content={"error": "No filename provided"},
                status_code=400
            )
        
        # Read audio
        audio_bytes = await file.read()
        
        if len(audio_bytes) == 0:
            return JSONResponse(
                content={"error": "Empty file received"},
                status_code=400
            )
        
        # Load audio with librosa
        audio_stream = io.BytesIO(audio_bytes)
        audio, sample_rate = librosa.load(audio_stream, sr=None)
        
        # Extract features (shape: 1, 39, 1)
        features_array = extract_features_from_audio(audio, sample_rate)
        
        # Flatten for JSON response
        features_flat = features_array.flatten().tolist()
        
        # Prepare response
        response = {
            "success": True,
            "filename": file.filename,
            "audio_info": {
                "duration_seconds": round(len(audio) / sample_rate, 3),
                "sample_rate": sample_rate,
                "samples": len(audio),
                "channels": 1  # librosa loads mono
            },
            "features": {
                "shape": features_array.shape,  # [1, 39, 1]
                "count": 39,
                "array_3d": features_array.tolist(),  # [[[...]]]
                "flat_39": features_flat,  # [0.1, -0.2, ...]
                "breakdown": {
                    "mfccs_13": features_flat[0:13],
                    "deltas_13": features_flat[13:26],
                    "delta_deltas_13": features_flat[26:39]
                }
            }
        }
        
        print(f"✅ Successfully extracted features from {file.filename}")
        return response
        
    except librosa.LibrosaError as e:
        print(f"Librosa error: {e}")
        return JSONResponse(
            content={"error": f"Audio processing error: {str(e)}"},
            status_code=400
        )
    except Exception as e:
        print(f"Unexpected error: {e}")
        return JSONResponse(
            content={"error": f"Server error: {str(e)}"},
            status_code=500
        )