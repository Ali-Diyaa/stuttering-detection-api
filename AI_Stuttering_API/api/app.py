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

# Remove all cleanup code - Koyeb handles this
# Delete lines 12-40 (the cleanup function and its call)

# Load models
import os
import tensorflow as tf

# Get the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

# Debug: Print paths
print(f"BASE_DIR: {BASE_DIR}")
print(f"MODELS_DIR: {MODELS_DIR}")
print(f"MODELS_DIR exists: {os.path.exists(MODELS_DIR)}")

# List files in models directory
if os.path.exists(MODELS_DIR):
    print("Files in models directory:")
    for file in os.listdir(MODELS_DIR):
        print(f"  - {file}")

model_paths = {
    'Prolongation': os.path.join(MODELS_DIR, 'model_Prolongation_3.h5'),
    'Block': os.path.join(MODELS_DIR, 'model_Block_3.h5'),
    'SoundRep': os.path.join(MODELS_DIR, 'model_SoundRep_3.h5'),
    'WordRep': os.path.join(MODELS_DIR, 'model_WordRep_3.h5'),
    'Interjection': os.path.join(MODELS_DIR, 'model_Interjection_3.h5')
}

# OR if using original names with spaces:
# model_paths = {
#     'Prolongation': os.path.join(MODELS_DIR, 'model_Prolongation (3).h5'),
#     'Block': os.path.join(MODELS_DIR, 'model_Block (3).h5'),
#     'SoundRep': os.path.join(MODELS_DIR, 'model_SoundRep (3).h5'),
#     'WordRep': os.path.join(MODELS_DIR, 'model_WordRep (3).h5'),
#     'Interjection': os.path.join(MODELS_DIR, 'model_Interjection (3).h5')
# }

# Check each file exists
for name, path in model_paths.items():
    if os.path.exists(path):
        print(f"✅ Model {name} found at: {path}")
    else:
        print(f"❌ Model {name} NOT found at: {path}")



def safe_load_model(model_path):
    """Load model with compatibility fixes"""
    try:
        # Try normal load first
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"Normal load failed: {e}")
        
        try:
            # Try custom_objects approach
            model = tf.keras.models.load_model(
                model_path,
                custom_objects={
                    'batch_shape': None,
                    'batch_input_shape': None
                },
                compile=False
            )
            return model
        except:
            # Try rebuilding from config
            with h5py.File(model_path, 'r') as f:
                config = f.attrs.get('model_config')
                if config:
                    config = config.decode('utf-8') if isinstance(config, bytes) else config
                    # Fix the config
                    config = config.replace('"batch_shape":', '"batch_input_shape":')
                    model = tf.keras.models.model_from_json(config)
                    
                    # Load weights
                    model.load_weights(model_path)
                    return model
        raise Exception(f"Could not load model: {model_path}")

# Then update your model loading:
models_dict = {}
for label, path in model_paths.items():
    try:
        model = safe_load_model(path)
        models_dict[label] = model
        print(f"✅ Loaded {label}")
    except Exception as e:
        print(f"❌ Failed to load {label}: {e}")



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
    return {
        "status": "healthy",
        "models_loaded": len(models_dict),
        "memory": "ok"
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