"""
SentimemeNet Backend API
Flask server for meme analysis using 6 deep learning models
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import base64
import io
import os
import re
from PIL import Image
import time

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)  # Enable CORS for all routes

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Model paths
MODEL_PATHS = {
    'meme_detector': os.path.join(MODELS_DIR, 'meme_detector_model.h5'),
    'humour': os.path.join(MODELS_DIR, 'meme_humour_model.h5'),
    'motivational': os.path.join(MODELS_DIR, 'meme_motivational_model.h5'),
    'offensive': os.path.join(MODELS_DIR, 'meme_offensive_model.h5'),
    'sarcasm': os.path.join(MODELS_DIR, 'meme_sarcasm_model.h5'),
    'sentiment': os.path.join(MODELS_DIR, 'meme_sentiment_model.h5'),
}

# Tokenizer paths
TOKENIZER_PATHS = {
    'humour': os.path.join(MODELS_DIR, 'tokenizer_humour.pickle'),
    'motivational': os.path.join(MODELS_DIR, 'tokenizer_motivational.pickle'),
    'offensive': os.path.join(MODELS_DIR, 'tokenizer_offensive.pickle'),
    'sarcasm': os.path.join(MODELS_DIR, 'tokenizer_sarcasm.pickle'),
    'sentiment': os.path.join(MODELS_DIR, 'tokenizer_sentiment.pickle'),
}

# Configuration
IMG_SIZE = (224, 224)
MAX_SEQUENCE_LENGTH = 100

# Custom layer for multimodal models
class NotEqual(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        new_args = []
        for a in args:
            try:
                if not tf.is_tensor(a):
                    a = tf.convert_to_tensor(a)
            except Exception:
                pass
            new_args.append(a)
        return super().__call__(*new_args, **kwargs)

    def call(self, *inputs, **kwargs):
        if len(inputs) == 1:
            single = inputs[0]
            if isinstance(single, (list, tuple)) and len(single) >= 2:
                x, y = single[0], single[1]
            else:
                raise ValueError("NotEqual layer received a single non-iterable input during call")
        else:
            x, y = inputs[0], inputs[1]

        try:
            if not tf.is_tensor(x):
                x = tf.convert_to_tensor(x)
        except Exception:
            pass
        try:
            if not tf.is_tensor(y):
                y = tf.convert_to_tensor(y)
        except Exception:
            pass

        try:
            x_dtype = getattr(x, 'dtype', None)
            y_dtype = getattr(y, 'dtype', None)
            if x_dtype is not None and y_dtype is not None and x_dtype != y_dtype:
                y = tf.cast(y, x_dtype)
        except Exception:
            pass

        return tf.math.not_equal(x, y)

    def get_config(self):
        config = super().get_config()
        return config


# Load models and tokenizers
print("[INFO] Loading models...")
models = {}
tokenizers = {}

try:
    # Load meme detector (no custom objects needed)
    models['meme_detector'] = tf.keras.models.load_model(MODEL_PATHS['meme_detector'])
    print(f"✓ Loaded meme_detector model")
    
    # Load multimodal models with custom objects
    custom_objects = {'NotEqual': NotEqual}
    for model_name in ['humour', 'motivational', 'offensive', 'sarcasm', 'sentiment']:
        models[model_name] = tf.keras.models.load_model(
            MODEL_PATHS[model_name], 
            custom_objects=custom_objects,
            compile=True
        )
        print(f"✓ Loaded {model_name} model")
        
        # Load tokenizer
        with open(TOKENIZER_PATHS[model_name], 'rb') as handle:
            tokenizers[model_name] = pickle.load(handle)
        print(f"✓ Loaded {model_name} tokenizer")
    
    print("[INFO] All models loaded successfully!")
    
except Exception as e:
    print(f"[ERROR] Failed to load models: {str(e)}")
    raise


def clean_text(text):
    """Clean and preprocess text"""
    if not text:
        return ''
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = text.strip()
    return text


def preprocess_image(image_data):
    """Preprocess image for model input"""
    # Decode base64 image
    img_bytes = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(img_bytes))
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize
    img = img.resize(IMG_SIZE)
    
    # Convert to array and normalize
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def preprocess_text(text, tokenizer):
    """Preprocess text for model input"""
    text_clean = clean_text(text)
    if text_clean == '':
        text_clean = 'no text available'
    
    text_sequence = tokenizer.texts_to_sequences([text_clean])
    text_padded = pad_sequences(
        text_sequence, 
        maxlen=MAX_SEQUENCE_LENGTH,
        padding='post', 
        truncating='post'
    )
    
    return text_padded


def predict_meme_detection(img_array):
    """Predict if image is a meme"""
    prediction = models['meme_detector'].predict(img_array, verbose=0)[0, 0]
    
    # prediction < 0.5 means Meme (class 0)
    # prediction >= 0.5 means Non-Meme (class 1)
    is_meme = prediction < 0.5
    confidence = (1 - prediction) if is_meme else prediction
    label = "Meme" if is_meme else "Non-Meme"
    
    return {
        'is_meme': bool(is_meme),
        'confidence': float(confidence),
        'label': label
    }


def predict_multimodal(model_name, img_array, text_padded, positive_label, negative_label):
    """Predict using multimodal model"""
    prediction = models[model_name].predict(
        [img_array, text_padded], 
        verbose=0
    )[0, 0]
    
    is_positive = prediction >= 0.5
    confidence = prediction if is_positive else (1 - prediction)
    label = positive_label if is_positive else negative_label
    
    return {
        'prediction': bool(is_positive),
        'confidence': float(confidence),
        'label': label
    }


@app.route('/')
def index():
    """Serve the frontend"""
    return send_from_directory('.', 'index.html')


@app.route('/api/analyze_meme', methods=['POST'])
def analyze_meme():
    """Main API endpoint for meme analysis"""
    start_time = time.time()
    
    try:
        # Get request data
        data = request.json
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        image_data = data.get('image')
        ocr_text = data.get('ocr_text', '')
        
        if not image_data:
            return jsonify({
                'success': False,
                'error': 'No image provided'
            }), 400
        
        # Preprocess image
        img_array = preprocess_image(image_data)
        
        # 1. Meme Detection
        meme_detection = predict_meme_detection(img_array)
        
        # If not a meme, skip sentiment analysis
        if not meme_detection['is_meme']:
            processing_time = time.time() - start_time
            return jsonify({
                'success': True,
                'results': {
                    'meme_detection': meme_detection,
                    'ocr_text': ocr_text,
                    'processing_time': f"{processing_time:.1f}s"
                }
            })
        
        # 2. Sentiment Analysis (all 5 models)
        results = {
            'meme_detection': meme_detection
        }
        
        # Preprocess text once for all models
        categories = {
            'humour': ('Funny', 'Not Funny'),
            'motivational': ('Motivational', 'Not Motivational'),
            'offensive': ('Offensive', 'Not Offensive'),
            'sarcasm': ('Sarcastic', 'Not Sarcastic'),
            'sentiment': ('Positive', 'Non-Positive')
        }
        
        for category, (pos_label, neg_label) in categories.items():
            text_padded = preprocess_text(ocr_text, tokenizers[category])
            results[category] = predict_multimodal(
                category, 
                img_array, 
                text_padded, 
                pos_label, 
                neg_label
            )
        
        # Add metadata
        results['ocr_text'] = ocr_text
        processing_time = time.time() - start_time
        results['processing_time'] = f"{processing_time:.1f}s"
        
        return jsonify({
            'success': True,
            'results': results
        })
    
    except Exception as e:
        print(f"[ERROR] Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(models),
        'tokenizers_loaded': len(tokenizers)
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("SentimemeNet Backend Server")
    print("="*60)
    print(f"Models loaded: {len(models)}")
    print(f"Tokenizers loaded: {len(tokenizers)}")
    print("="*60)
    print("\nStarting server on http://localhost:5000")
    print("Frontend available at: http://localhost:5000")
    print("API endpoint: http://localhost:5000/api/analyze_meme")
    print("\nPress CTRL+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
