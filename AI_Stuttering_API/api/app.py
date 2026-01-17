import tensorflow as tf
import librosa
import numpy as np
import io
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import traceback
from datetime import datetime
import tensorflow as tf
import h5py
import numpy as np
import threading

# Remove all cleanup code - Koyeb handles this
# Delete lines 12-40 (the cleanup function and its call)

# Load models
import os
import tensorflow as tf

# Custom InputLayer to handle batch_shape vs batch_input_shape
models_dict = {}
models_loaded = False
models_lock = threading.Lock()

def load_model_safe(model_path):
    """Safely load a single model with timeout"""
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        print(f"Failed to load {model_path}: {e}")
        return None

def load_models_background():
    """Load models in background thread"""
    global models_dict, models_loaded
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
    
    model_paths = {
        'Prolongation': os.path.join(MODELS_DIR, 'model_Prolongation_3.h5'),
        'Block': os.path.join(MODELS_DIR, 'model_Block_3.h5'),
        'SoundRep': os.path.join(MODELS_DIR, 'model_SoundRep_3.h5'),
        'WordRep': os.path.join(MODELS_DIR, 'model_WordRep_3.h5'),
        'Interjection': os.path.join(MODELS_DIR, 'model_Interjection_3.h5')
    }
    
    print("🚀 Starting background model loading...")
    
    # Load one model at a time to avoid memory issues
    for label, path in model_paths.items():
        print(f"Loading {label}...")
        model = load_model_safe(path)
        if model:
            with models_lock:
                models_dict[label] = model
            print(f"✅ {label} loaded")
        else:
            print(f"❌ {label} failed to load")
    
    models_loaded = True
    print("🎉 All models loaded!")

# Start background loading in a separate thread
import threading
threading.Thread(target=load_models_background, daemon=True).start()

# Feature extraction function (keep as is)
def extract_features_from_audio(audio, sr):
    try:
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        mfcc_mean = np.mean(mfcc.T, axis=0)
        delta_mean = np.mean(delta.T, axis=0)
        delta2_mean = np.mean(delta2.T, axis=0)
        
        features = np.concatenate([mfcc_mean, delta_mean, delta2_mean])
        return features.reshape(1, -1, 1)
    except Exception as e:
        raise Exception(f"Error extracting features: {str(e)}")

# Create FastAPI App
app = FastAPI(
    title="Stuttering Detection API",
    description="API for detecting different types of stuttering in audio files",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/")
async def root():
    return {
        "status": "running",
        "service": "Stuttering Detection API",
        "models": list(models_dict.keys()),
        "timestamp": datetime.now().isoformat()
    }

# Health check for monitoring
@app.get("/health")
async def health():
    """Health check that indicates if models are loaded"""
    loaded_count = len([m for m in models_dict.values() if m is not None])
    return {
        "status": "loading" if not models_loaded else "healthy",
        "models_loaded": f"{loaded_count}/5",
        "memory": "ok",
        "message": "Models are loading in background" if not models_loaded else "Ready"
    }
# Main prediction endpoint (keep your existing code)

@app.post("/predict/")
async def predict_stuttering(file: UploadFile = File(...)):
    try:
        print(f"Processing file: {file.filename} ({file.content_type})")

        if file.size == 0:
            return JSONResponse(
                content={"error": "Empty file received"},
                status_code=400
            )

        audio_content = await file.read()

        if len(audio_content) == 0:
            return JSONResponse(
                content={"error": "File content is empty"},
                status_code=400
            )

        audio_stream = io.BytesIO(audio_content)

        try:
            audio, sample_rate = librosa.load(audio_stream, sr=None)
        except Exception as e:
            return JSONResponse(
                content={"error": f"Could not load audio file: {str(e)}"},
                status_code=400
            )

        if len(audio) == 0:
            return JSONResponse(
                content={"error": "Audio file contains no data"},
                status_code=400
            )

        try:
            features = extract_features_from_audio(audio, sample_rate)
        except Exception as e:
            return JSONResponse(
                content={"error": f"Feature extraction failed: {str(e)}"},
                status_code=400
            )

        predictions = {}
        detected_types = []

        for label, model in models_dict.items():
            try:
                prediction = model.predict(features, verbose=0)
                prob = float(prediction[0][0])
                predictions[label] = prob

                if prob > 0.75:
                    detected_types.append(label)

            except Exception as e:
                print(f"Error predicting with model {label}: {e}")
                predictions[label] = 0.0

        response = {
            "status": "success",
            "filename": file.filename,
            "audio_info": {
                "duration_seconds": round(len(audio) / sample_rate, 2),
                "sample_rate": sample_rate,
                "samples": len(audio)
            },
            "detected_types": detected_types if detected_types else ["Normal speech (no stuttering detected)"],
            "confidence_scores": predictions,
            "threshold": 0.7
        }

        print(f"Analysis complete for {file.filename}")
        return JSONResponse(content=response)

    except Exception as e:
        print(f"Error in predict_stuttering: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse(
            content={
                "error": f"Internal server error: {str(e)}",
                "traceback": traceback.format_exc()
            },
            status_code=500
        )