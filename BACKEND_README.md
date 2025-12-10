# SentimemeNet - Backend + Frontend Integration

Complete deep learning meme analysis system with Flask backend and HTML frontend.

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- TensorFlow 2.15.0
- All trained models in `models/` folder

### Installation & Running

**Option 1: PowerShell Script (Recommended)**
```powershell
.\start_server.ps1
```

**Option 2: Manual**
```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python app.py
```

The server will start on `http://localhost:5000` and automatically serve the frontend.

## 📁 Project Structure

```
SentimemeNet/
├── app.py                      # Flask backend server
├── requirements.txt            # Python dependencies
├── start_server.ps1           # Quick start script
├── models/                     # Trained models
│   ├── meme_detector_model.h5
│   ├── meme_humour_model.h5
│   ├── meme_motivational_model.h5
│   ├── meme_offensive_model.h5
│   ├── meme_sarcasm_model.h5
│   ├── meme_sentiment_model.h5
│   ├── tokenizer_humour.pickle
│   ├── tokenizer_motivational.pickle
│   ├── tokenizer_offensive.pickle
│   ├── tokenizer_sarcasm.pickle
│   └── tokenizer_sentiment.pickle
├── frontend/                   # HTML frontend
│   ├── index.html
│   ├── styles.css
│   ├── script.js
│   └── README.md
└── notebooks/                  # Training notebooks
    └── ...
```

## 🎯 Features

### Backend (Flask API)
- ✅ REST API endpoint: `/api/analyze_meme`
- ✅ Health check endpoint: `/api/health`
- ✅ Automatic model loading on startup
- ✅ Image preprocessing (base64 decode, resize, normalize)
- ✅ Text preprocessing (cleaning, tokenization, padding)
- ✅ Meme detection using Mini-ResNet CNN
- ✅ 5 sentiment models using multimodal CNN+LSTM
- ✅ CORS enabled for frontend integration
- ✅ Error handling and validation

### Frontend (HTML/JS)
- ✅ Drag-and-drop image upload
- ✅ File browse and clipboard paste
- ✅ Optional OCR text input
- ✅ Real-time analysis with loading states
- ✅ Results display with confidence scores
- ✅ Export results to JSON
- ✅ Responsive design
- ✅ Sample meme loader

## 🔧 API Documentation

### Analyze Meme
**Endpoint:** `POST /api/analyze_meme`

**Request:**
```json
{
  "image": "base64_encoded_image_string",
  "ocr_text": "optional text from meme"
}
```

**Response (Meme Detected):**
```json
{
  "success": true,
  "results": {
    "meme_detection": {
      "is_meme": true,
      "confidence": 0.9253,
      "label": "Meme"
    },
    "humour": {
      "prediction": true,
      "confidence": 0.5220,
      "label": "Funny"
    },
    "motivational": {
      "prediction": false,
      "confidence": 0.6155,
      "label": "Not Motivational"
    },
    "offensive": {
      "prediction": false,
      "confidence": 0.5858,
      "label": "Not Offensive"
    },
    "sarcasm": {
      "prediction": true,
      "confidence": 0.5585,
      "label": "Sarcastic"
    },
    "sentiment": {
      "prediction": true,
      "confidence": 0.5810,
      "label": "Positive"
    },
    "ocr_text": "text from meme",
    "processing_time": "4.2s"
  }
}
```

**Response (Not a Meme):**
```json
{
  "success": true,
  "results": {
    "meme_detection": {
      "is_meme": false,
      "confidence": 0.8745,
      "label": "Non-Meme"
    },
    "ocr_text": "",
    "processing_time": "1.2s"
  }
}
```

**Error Response:**
```json
{
  "success": false,
  "error": "Error message"
}
```

### Health Check
**Endpoint:** `GET /api/health`

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": 6,
  "tokenizers_loaded": 5
}
```

## 🧠 Model Details

### 1. Meme Detection (92.53% accuracy)
- **Architecture:** Mini-ResNet CNN (6 residual blocks)
- **Input:** 224×224 RGB images
- **Output:** Binary (Meme/Non-Meme)

### 2. Sentiment Models (Multimodal CNN+LSTM)
- **Humor** (52.20% accuracy) - Funny/Not Funny
- **Motivational** (61.55% accuracy) - Motivational/Not Motivational
- **Offensive** (58.58% accuracy) - Offensive/Not Offensive
- **Sarcasm** (55.85% accuracy) - Sarcastic/Not Sarcastic
- **Sentiment** (58.10% accuracy) - Positive/Non-Positive

All sentiment models use:
- **Image Branch:** ResNet-based CNN
- **Text Branch:** Bidirectional LSTM
- **Input:** 224×224 images + OCR text (max 100 tokens)

## 🔍 Usage Examples

### Using the Web Interface
1. Open `http://localhost:5000` in your browser
2. Upload a meme image
3. Optionally add text from the meme
4. Click "Analyze Meme"
5. View results and export if needed

### Using cURL
```bash
# Convert image to base64
base64_image=$(base64 -w 0 your_meme.jpg)

# Make API request
curl -X POST http://localhost:5000/api/analyze_meme \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$base64_image\", \"ocr_text\": \"Your meme text\"}"
```

### Using Python
```python
import requests
import base64

# Read and encode image
with open('meme.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

# Make request
response = requests.post('http://localhost:5000/api/analyze_meme', json={
    'image': image_data,
    'ocr_text': 'Optional text from meme'
})

results = response.json()
print(results)
```

## ⚙️ Configuration

### Server Settings
Edit `app.py` to change:
- **Port:** Default is `5000`
- **Host:** Default is `0.0.0.0` (all interfaces)
- **Debug:** Set to `False` for production

### Model Settings
- **Image Size:** `IMG_SIZE = (224, 224)`
- **Max Text Length:** `MAX_SEQUENCE_LENGTH = 100`

## 🐛 Troubleshooting

### Models not loading
- Ensure all `.h5` files are in `models/` folder
- Check TensorFlow version compatibility
- Verify tokenizer `.pickle` files exist

### Port already in use
```powershell
# Change port in app.py
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Frontend not connecting
- Verify backend is running on `http://localhost:5000`
- Check browser console for CORS errors
- Ensure `flask-cors` is installed

### Slow predictions
- First prediction may be slow (model initialization)
- Subsequent predictions are faster
- Consider using GPU acceleration

## 📊 Performance

- **Meme Detection:** ~1-2 seconds
- **Full Analysis (all 6 models):** ~4-6 seconds
- **Image Preprocessing:** <0.5 seconds

Times may vary based on:
- Image size
- CPU/GPU availability
- System resources

## 🔐 Security Notes

For production deployment:
- ✅ Add rate limiting
- ✅ Implement authentication
- ✅ Validate image sizes and formats
- ✅ Set up HTTPS
- ✅ Configure proper CORS origins
- ✅ Add input sanitization
- ✅ Use environment variables for config

## 📝 License

CCS 248 Final Project - SentimemeNet

## 🙏 Acknowledgments

- TensorFlow/Keras for deep learning framework
- Flask for web framework
- Memotion Dataset for training data
- Caltech 256 for non-meme images

## 📧 Support

For issues or questions:
1. Check console logs for errors
2. Verify all dependencies are installed
3. Ensure models are in correct location
4. Review API documentation
