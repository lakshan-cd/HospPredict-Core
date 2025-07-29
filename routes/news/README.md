# News Prediction Routes

This module contains the news-based stock price prediction API routes for the Hotel Financial Risk Predictive API.

## 📁 Structure

```
routes/news/
├── __init__.py           # Module initialization
├── prediction.py         # Core prediction endpoints
├── analytics.py          # Analytics and insights
├── admin.py             # Admin and management tools
├── websocket.py         # Real-time WebSocket support
├── test_news_api.py     # Test script
└── README.md            # This file
```

## 🚀 Features

### Core Prediction
- **Single Prediction**: Predict for individual news articles
- **Batch Prediction**: Process multiple articles simultaneously
- **File Upload**: Bulk predictions from CSV files
- **Automatic Feature Engineering**: Creates 20+ features from minimal input

### Analytics
- **Performance Metrics**: Model accuracy and performance
- **Prediction Trends**: Time-series analysis
- **Hotel Insights**: Hotel-specific analytics
- **Sentiment Analysis**: News sentiment trends
- **Source Credibility**: News source impact analysis

### Admin Tools
- **System Monitoring**: Health checks and status
- **Model Management**: Backup, restore, validation
- **Logging**: System logs and debugging
- **Configuration**: System settings

### Real-time Features
- **WebSocket Support**: Live updates and notifications
- **Real-time Predictions**: Instant processing
- **Live Analytics**: Real-time insights
- **Alert System**: System notifications

## 🔗 API Endpoints

### Core Prediction Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/news/` | GET | Root endpoint with API info |
| `/api/v1/news/health` | GET | Health check |
| `/api/v1/news/predict` | POST | Single article prediction |
| `/api/v1/news/predict/batch` | POST | Batch prediction |
| `/api/v1/news/predict/upload` | POST | File upload for predictions |
| `/api/v1/news/hotels` | GET | List supported hotels |
| `/api/v1/news/features` | GET | Feature information |
| `/api/v1/news/models/info` | GET | Model information |
| `/api/v1/news/models/reload` | POST | Reload models |

### Analytics Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/news/analytics/performance` | GET | Model performance metrics |
| `/api/v1/news/analytics/trends` | GET | Prediction trends |
| `/api/v1/news/analytics/hotels/insights` | GET | Hotel insights |
| `/api/v1/news/analytics/sentiment/analysis` | GET | Sentiment analysis |
| `/api/v1/news/analytics/sources/credibility` | GET | Source credibility |
| `/api/v1/news/analytics/predictions/summary` | GET | Prediction summary |

### Admin Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/news/admin/system/info` | GET | System information |
| `/api/v1/news/admin/models/status` | GET | Model status |
| `/api/v1/news/admin/models/backup` | POST | Create backup |
| `/api/v1/news/admin/backups` | GET | List backups |
| `/api/v1/news/admin/models/restore/{id}` | POST | Restore models |
| `/api/v1/news/admin/models/validate` | POST | Validate models |
| `/api/v1/news/admin/logs` | GET | System logs |

### WebSocket Endpoints

| Endpoint | Description |
|----------|-------------|
| `/api/v1/news/ws/` | Main WebSocket |
| `/api/v1/news/ws/predictions` | Prediction updates |
| `/api/v1/news/ws/alerts` | System alerts |
| `/api/v1/news/ws/analytics` | Analytics updates |

## 📝 Usage Examples

### Single Prediction

```python
import requests

# Single prediction
response = requests.post("http://localhost:8000/api/v1/news/predict", json={
    "date": "2024-01-15",
    "heading": "Tourism surge expected after visa reforms",
    "text": "Sri Lanka tourism is expected to see a massive surge...",
    "source": "economynext"
})

result = response.json()
print(f"Spike: {result['spike_prediction']}")
print(f"Hotels: {len(result['hotel_predictions'])} predictions")
```

### Batch Prediction

```python
# Batch prediction
articles = [
    {
        "date": "2024-01-15",
        "heading": "Tourism surge expected",
        "text": "Sri Lanka tourism is expected to see a massive surge...",
        "source": "economynext"
    },
    {
        "date": "2024-01-15",
        "heading": "Hotel profits decline",
        "text": "Major hotels report declining profits...",
        "source": "dailynews"
    }
]

response = requests.post("http://localhost:8000/api/v1/news/predict/batch", json={
    "articles": articles
})

result = response.json()
print(f"Processed: {result['summary']['successful_predictions']} articles")
```

### WebSocket Connection

```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/news/ws/');

ws.onopen = function() {
    // Subscribe to predictions
    ws.send(JSON.stringify({
        type: 'subscribe',
        topic: 'predictions'
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
};
```

## 🏗️ Model Requirements

The API expects the following model files in `models/news/models/`:

```
models/news/models/
├── spike_model.pkl          # Spike prediction model
├── vectorizer.pkl           # TF-IDF vectorizer
├── label_encoders.pkl       # Label encoders for hotels
└── Price_Models/
    ├── price_model_AHUN.N0000.pkl
    ├── price_model_TRAN.N0000.pkl
    ├── price_model_STAF.N0000.pkl
    └── ... (20 hotel models)
```

## 🔧 Input Requirements

### Required Fields
- **date**: Article date (YYYY-MM-DD format)
- **heading**: News headline
- **text**: Article content
- **source**: News source

### Supported Sources
- economynext
- dailynews
- sundaytimes
- island
- colombopage
- newsfirst
- adaderana
- hirunews
- lankadeepa

## 📊 Output Format

### Single Prediction Response

```json
{
    "article_id": "article_20240115_143022",
    "date": "2024-01-15",
    "heading": "Tourism surge expected after visa reforms",
    "source": "economynext",
    "spike_prediction": "SPIKE",
    "spike_confidence": 0.75,
    "hotel_predictions": {
        "AHUN.N0000": {
            "prediction": "up",
            "confidence": 0.82,
            "probabilities": {
                "up": 0.82,
                "down": 0.08,
                "neutral": 0.10
            }
        }
    }
}
```

## 🧪 Testing

Run the test script to verify the API is working:

```bash
cd routes/news
python test_news_api.py
```

This will test all endpoints and provide a comprehensive report.

## 🔍 Monitoring

### Health Checks
- `/api/v1/news/health` - Basic health check
- `/api/v1/news/admin/system/info` - Detailed system info
- `/api/v1/news/admin/models/status` - Model status

### Logging
The API provides comprehensive logging for:
- Request processing
- Model loading
- Error handling
- Performance metrics

## 🚀 Integration

The news prediction routes are integrated into the main API at:

- **Base URL**: `/api/v1/news/`
- **Documentation**: Available at `/docs` when the main API is running
- **Health Check**: `/api/v1/news/health`

## 📈 Performance

### Expected Response Times
- **Single prediction**: < 2 seconds
- **Batch prediction**: < 5 seconds per article
- **File upload**: Depends on file size

### Scalability
- Stateless design
- Horizontal scaling support
- Background task processing
- Caching capabilities

## 🔒 Security

### Production Considerations
1. **Authentication**: Implement JWT-based authentication
2. **Rate Limiting**: Add rate limiting middleware
3. **HTTPS**: Use SSL/TLS certificates
4. **Input Validation**: Validate all inputs
5. **Model Security**: Secure model files

## 📞 Support

For issues and questions:
- Check the API documentation at `/docs`
- Review the test script for examples
- Check the logs for error details
- Verify model files are in the correct location 