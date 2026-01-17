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

# Custom InputLayer to handle batch_shape vs batch_input_shape
class CustomInputLayer(tf.keras.layers.InputLayer):
    """Custom InputLayer to handle batch_shape parameter"""
    def __init__(self, **kwargs):
        # Convert batch_shape to batch_input_shape
        if 'batch_shape' in kwargs:
            kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
        super().__init__(**kwargs)

def load_model_with_fix(model_path):
    """Load model with compatibility fixes for batch_shape issue"""
    try:
        # Method 1: Try normal load
        print(f"Attempting normal load for {model_path}")
        model = tf.keras.models.load_model(model_path)
        print("✅ Success with normal load")
        return model
    except Exception as e1:
        print(f"❌ Normal load failed: {str(e1)[:100]}")
        
        try:
            # Method 2: Try with custom objects
            print("Attempting load with custom objects...")
            model = tf.keras.models.load_model(
                model_path,
                custom_objects={'InputLayer': CustomInputLayer},
                compile=False
            )
            print("✅ Success with custom objects")
            return model
        except Exception as e2:
            print(f"❌ Custom objects failed: {str(e2)[:100]}")
            
            try:
                # Method 3: Try to fix the H5 file
                print("Attempting to fix H5 file...")
                with h5py.File(model_path, 'r') as f:
                    if 'model_config' in f.attrs:
                        config = f.attrs['model_config']
                        if isinstance(config, bytes):
                            config = config.decode('utf-8')
                        
                        # Fix batch_shape -> batch_input_shape
                        config = config.replace('"batch_shape":', '"batch_input_shape":')
                        
                        # Create model from fixed config
                        model = tf.keras.models.model_from_json(config)
                        
                        # Load weights
                        model.load_weights(model_path)
                        print("✅ Success by fixing H5 file")
                        return model
            except Exception as e3:
                print(f"❌ All methods failed: {str(e3)[:100]}")
                raise Exception(f"Could not load model: {model_path}")

# Get paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

model_paths = {
    'Prolongation': os.path.join(MODELS_DIR, 'model_Prolongation_3.h5'),
    'Block': os.path.join(MODELS_DIR, 'model_Block_3.h5'),
    'SoundRep': os.path.join(MODELS_DIR, 'model_SoundRep_3.h5'),
    'WordRep': os.path.join(MODELS_DIR, 'model_WordRep_3.h5'),
    'Interjection': os.path.join(MODELS_DIR, 'model_Interjection_3.h5')
}

# Load models
models_dict = {}
for label, path in model_paths.items():
    if os.path.exists(path):
        print(f"\n=== Loading {label} ===")
        try:
            model = load_model_with_fix(path)
            models_dict[label] = model
            print(f"✅ Successfully loaded {label}")
        except Exception as e:
            print(f"❌ Failed to load {label}: {str(e)[:200]}")
            models_dict[label] = None
    else:
        print(f"❌ Model file not found: {path}")
        models_dict[label] = None

print(f"\n=== Summary ===")
print(f"Total models: {len(model_paths)}")
print(f"Successfully loaded: {sum(1 for m in models_dict.values() if m is not None)}")
print(f"Failed: {sum(1 for m in models_dict.values() if m is None)}")
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