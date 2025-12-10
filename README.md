# SentimemeNet Frontend

Modern, responsive HTML frontend for the SentimemeNet deep learning meme analysis system.

## Overview

This frontend provides a user-friendly interface for analyzing memes using 6 specialized deep learning models:
1. **Meme Detection** - Identifies if an image is a meme (92.53% accuracy)
2. **Humor Classification** - Determines if a meme is funny
3. **Motivational Classification** - Detects motivational content
4. **Offensive Classification** - Identifies offensive content
5. **Sarcasm Classification** - Detects sarcastic elements
6. **Sentiment Classification** - Analyzes overall sentiment

## Features

### Core Functionality
- ✅ Drag-and-drop image upload
- ✅ File browse upload (JPG, PNG, GIF)
- ✅ Clipboard paste support
- ✅ Image preview with remove option
- ✅ Optional OCR text input (500 char limit)
- ✅ Real-time character counter
- ✅ Sample meme loader
- ✅ Analysis button with loading states
- ✅ Results display with confidence scores
- ✅ Progress bars and visual indicators
- ✅ Export results to JSON
- ✅ Processing time display
- ✅ Model performance information
- ✅ Fully responsive design

### User Experience
- Clean, modern card-based design
- Smooth animations and transitions
- Color-coded results (green=positive, red=negative/offensive, blue/purple/yellow for categories)
- Empty state guidance
- Error handling and validation
- Mobile-friendly interface

## Files Structure

```
frontend/
├── index.html          # Main HTML structure
├── styles.css          # Custom CSS styles
├── script.js           # JavaScript functionality
└── README.md           # This file
```

## Setup Instructions

### 1. Basic Setup (No Backend)
Simply open `index.html` in a web browser. The frontend will work with mock data for demonstration purposes.

```bash
# Open directly
open index.html

# Or use a simple HTTP server
python -m http.server 8000
# Then visit: http://localhost:8000
```

### 2. Backend Integration

To connect to the actual SentimemeNet backend, you need to implement a backend API endpoint.

#### API Endpoint Required

**POST** `/api/analyze_meme`

**Request Format:**
```json
{
  "image": "base64_encoded_image_string",
  "ocr_text": "optional manual text" or null
}
```

**Response Format:**
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
    "ocr_text": "extracted or provided text",
    "processing_time": "4.2s"
  }
}
```

**Error Response:**
```json
{
  "success": false,
  "error": "Error message description"
}
```

### 3. Python Flask Backend Example

Create a simple Flask backend (`app.py`):

```python
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import base64
from PIL import Image
import io

app = Flask(__name__)

# Load your models
meme_detector = tf.keras.models.load_model('models/meme_detector_model.h5')
# ... load other models

@app.route('/api/analyze_meme', methods=['POST'])
def analyze_meme():
    try:
        data = request.json
        image_data = base64.b64decode(data['image'])
        ocr_text = data.get('ocr_text', '')
        
        # Process image
        image = Image.open(io.BytesIO(image_data))
        # ... preprocess and run through models
        
        results = {
            'success': True,
            'results': {
                # Your model predictions here
            }
        }
        return jsonify(results)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
```

Run the backend:
```bash
python app.py
```

Then update `script.js` to point to `http://localhost:5000/api/analyze_meme`

## Customization

### Colors
Edit the Tailwind config in `index.html`:
```javascript
tailwind.config = {
    theme: {
        extend: {
            colors: {
                primary: '#4A90E2',    // Blue
                secondary: '#7ED321',   // Green
                danger: '#D0021B',      // Red
                warning: '#F5A623',     // Orange
                success: '#7ED321'      // Green
            }
        }
    }
}
```

### Model Information
Update the model performance cards in `index.html` with your actual model metrics.

### API Endpoint
Change the API endpoint in `script.js`:
```javascript
const response = await fetch('YOUR_API_ENDPOINT_HERE', {
    method: 'POST',
    // ...
});
```

## Browser Compatibility

- ✅ Chrome/Edge (latest)
- ✅ Firefox (latest)
- ✅ Safari (latest)
- ✅ Mobile browsers (iOS Safari, Chrome Mobile)

## Technologies Used

- **HTML5** - Structure
- **CSS3** - Styling with custom animations
- **Tailwind CSS** - Utility-first CSS framework (CDN)
- **JavaScript (ES6+)** - Functionality
- **Feather Icons** - Icon library
- **Fetch API** - HTTP requests

## Validation & Security

### Client-Side Validation
- File type validation (JPEG, PNG, GIF only)
- File size limit (10MB max)
- Text length limit (500 characters)
- Image preview before upload

### Security Considerations
- Uses HTTPS for API calls (configure in production)
- No sensitive data stored in localStorage
- XSS prevention through proper escaping
- CORS configuration needed for production

## Development Tips

### Testing Without Backend
The frontend includes mock data generation for testing without a backend. The `generateMockResults()` function provides realistic sample data.

### Debugging
Open browser DevTools Console to see:
- API call logs
- Error messages
- State changes

### Performance
- Images are base64 encoded before sending (consider optimization for large files)
- Lazy loading of results with staggered animations
- Debounced text input for character counting

## Future Enhancements

Potential features to add:
- [ ] Batch upload (multiple memes)
- [ ] History of recent analyses (localStorage)
- [ ] Dark mode toggle
- [ ] Real OCR integration (Tesseract.js)
- [ ] CSV export option
- [ ] Comparison view (side-by-side memes)
- [ ] Social media sharing
- [ ] Progressive Web App (PWA) support
- [ ] Webcam capture

## Troubleshooting

### Issue: Images not uploading
- Check file size (<10MB)
- Verify file type (JPEG, PNG, GIF)
- Check browser console for errors

### Issue: Analysis not working
- Verify backend is running
- Check API endpoint URL
- Review CORS settings
- Check network tab in DevTools

### Issue: Results not displaying
- Verify API response format matches expected structure
- Check console for JavaScript errors
- Ensure all DOM elements have correct IDs

## License

This frontend is part of the SentimemeNet project (CCS 248 Final Project).

## Support

For issues or questions:
1. Check the browser console for errors
2. Verify API endpoint connectivity
3. Review the backend logs
4. Ensure all model files are properly loaded

## Credits

- **Project**: SentimemeNet
- **Course**: CCS 248
- **Framework**: TensorFlow/Keras
- **Frontend**: HTML5, CSS3, JavaScript
- **Icons**: Feather Icons
- **CSS Framework**: Tailwind CSS
