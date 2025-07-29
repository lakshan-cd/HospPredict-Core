from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime, timedelta
import logging
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings('ignore')

# Import VADER for sentiment analysis
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logging.warning("VADER sentiment analysis not available. Using simple keyword-based approach.")


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Initialize VADER analyzer
vader_analyzer = None
if VADER_AVAILABLE:
    try:
        vader_analyzer = SentimentIntensityAnalyzer()
        logger.info("✅ VADER sentiment analyzer initialized successfully")
    except Exception as e:
        logger.warning(f"⚠️ Failed to initialize VADER analyzer: {e}")
        VADER_AVAILABLE = False
else:
    logger.warning("⚠️ VADER not available, using simple keyword-based sentiment analysis")


# Initialize router
router = APIRouter(prefix="/news", tags=["news-prediction"])

# Global variables for models
models = {}
vectorizer = None
label_encoders = None
hotel_codes = []

# Pydantic models for request/response
class NewsInput(BaseModel):
    date: str = Field(..., description="Date of the news article (YYYY-MM-DD)")
    heading: str = Field(..., description="News headline")
    text: str = Field(..., description="News article content")
    source: str = Field(..., description="News source")

class NewsBatchInput(BaseModel):
    articles: List[NewsInput] = Field(..., description="List of news articles")

class PredictionResponse(BaseModel):
    article_id: str
    date: str
    heading: str
    source: str
    spike_prediction: str
    spike_confidence: float
    hotel_predictions: Dict[str, Dict[str, Any]]

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    summary: Dict[str, Any]

# Feature engineering functions
def create_temporal_features(date_str: str) -> Dict[str, Any]:
    """Create temporal features from date"""
    try:
        date = pd.to_datetime(date_str)
        return {
            'day_of_week': date.dayofweek,
            'month': date.month,
            'quarter': date.quarter,
            'year': date.year,
            'is_weekend': int(date.dayofweek in [5, 6]),
            'is_month_start': int(date.is_month_start),
            'is_month_end': int(date.is_month_end),
            'is_holiday_season': int(date.month in [12, 1, 2]),
            'is_peak_season': int(date.month in [3, 4, 7, 8, 12])
        }
    except:
        return {
            'day_of_week': 0, 'month': 1, 'quarter': 1, 'year': 2024,
            'is_weekend': 0, 'is_month_start': 0, 'is_month_end': 0,
            'is_holiday_season': 0, 'is_peak_season': 0
        }

def create_text_features(heading: str, text: str) -> Dict[str, Any]:
    """Create text-based features"""
    full_text = f"{heading} {text}"
    words = full_text.split()
    
    return {
        'text_length': len(full_text),
        'word_count': len(words),
        'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
        'text_complexity': len(set(words)) / len(words) if words else 0
    }

def get_sentiment_encoded(sentiment: str) -> int:
    """Convert sentiment to encoded value"""
    sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
    return sentiment_map.get(sentiment.lower(), 0)

def get_source_credibility(source: str) -> float:
    """Get source credibility score (matching training environment)"""
    source_credibility = {
        'economynext': 0.9,
        'dailynews': 0.8,
        'sundaytimes': 0.85,
        'island': 0.75,
        'colombopage': 0.7,
        'newsfirst': 0.8,
        'adaderana': 0.75,
        'hirunews': 0.7,
        'lankadeepa': 0.8,
        'ceylontoday': 0.7,
        'news.lk': 0.75,
        'srilankamirror': 0.65,
        'asianmirror': 0.7,
        'srilankaguardian': 0.75,
        'groundviews': 0.8,
        'colombotelegraph': 0.7,
        'srilankabrief': 0.65,
        'tamilguardian': 0.6,
        'jdslanka': 0.7,
        'srilankamirror': 0.65
    }
    return source_credibility.get(source.lower(), 0.5)

def classify_news_type_and_hotel(heading: str, text: str) -> tuple:
    """Classify news type and hotel (same as training code)"""
    full_text = f"{heading} {text}".lower()
    
    # Hotel aliases from training code
    hotel_aliases = {
        "AITKEN SPENCE": ["aitken", "spence"],
        "ASIAN HOTELS": ["asian hotels", "cinnamon red", "cinnamon grand", "cinnamon lakeside"],
        "BROWNS BEACH": ["browns beach"],
        "CEYLON HOTELS": ["ceylon hotels", "queens hotel", "galle face"],
        "DOLPHIN HOTELS": ["dolphin"],
        "EDEN HOTEL": ["eden"],
        "GALADARI HOTELS": ["galadari"],
        "HOTEL SIGIRIYA": ["hotel sigiriya", "sigiriya plc"],
        "JOHN KEELLS": ["john keells", "cinnamon", "hikka tranz"],
        "MAHAWELI REACH": ["mahaweli reach"],
        "PALM GARDEN": ["palm garden"],
        "PEGASUS HOTELS": ["pegasus"],
        "RENUKA CITY": ["renuka city"],
        "RENUKA HOTELS": ["renuka hotels"],
        "ROYAL PALMS": ["royal palms"],
        "SERENDIB HOTELS": ["serendib"],
        "SIGIRIYA VILLAGE": ["sigiriya village"],
        "TAL LANKA": ["tal lanka"],
        "TANGERINE BEACH": ["tangerine beach"],
        "LIGHTHOUSE HOTEL": ["lighthouse"],
        "KANDY HOTELS": ["kandy hotels", "queens hotel"],
        "TRANS ASIA": ["trans asia"],
    }
    
    # Hotel symbol mapping from training code
    hotel_symbol_map = {
        "AITKEN SPENCE": "AHUN.N0000",
        "ASIAN HOTELS": "AHPL.N0000",
        "BROWNS BEACH": "BBH.N0000",
        "CEYLON HOTELS": "CHOT.N0000",
        "DOLPHIN HOTELS": "STAF.N0000",
        "EDEN HOTEL": "EDEN.N0000",
        "GALADARI HOTELS": "GHLL.N0000",
        "HOTEL SIGIRIYA": "HSIG.N0000",
        "JOHN KEELLS": "KJL.N0000",
        "MAHAWELI REACH": "MRH.N0000",
        "PALM GARDEN": "PALM.N0000",
        "PEGASUS HOTELS": "PEG.N0000",
        "RENUKA CITY": "RENU.N0000",
        "RENUKA HOTELS": "RCH.N0000",
        "ROYAL PALMS": "RPBH.N0000",
        "SERENDIB HOTELS": "SHOT.N0000",
        "SIGIRIYA VILLAGE": "SIGV.N0000",
        "TAL LANKA": "TAJ.N0000",
        "TANGERINE BEACH": "TANG.N0000",
        "LIGHTHOUSE HOTEL": "LHL.N0000",
        "KANDY HOTELS": "KHC.N0000",
        "TRANS ASIA": "TRAN.N0000"
    }
    
    # Tourism keywords from training code
    tourism_keywords = [
        "tourism", "travel", "holiday", "resort", "hotel industry", "beach", "sri lanka tourism",
        "tourist", "vacation", "hospitality", "wildlife park", "ayurveda", "unawatuna", "ella",
        "sigiriya", "yala", "kandy", "galle fort", "ecotourism", "foreign arrivals"
    ]
    
    # Check for hotel-specific keywords first
    for hotel, keywords in hotel_aliases.items():
        for keyword in keywords:
            if keyword in full_text:
                return 'hotel', hotel_symbol_map.get(hotel, hotel)
    
    # Check for tourism keywords
    for keyword in tourism_keywords:
        if keyword in full_text:
            return 'tourism', 'general'
    
    # Default to general
    return 'general', 'general'

def get_sentiment(heading: str, text: str) -> str:
    """VADER sentiment analysis (same as training code)"""
    if not VADER_AVAILABLE or vader_analyzer is None:
        # Fallback to simple keyword-based approach
        full_text = f"{heading} {text}".lower()
        positive_words = ['profit', 'growth', 'increase', 'positive', 'success', 'record', 'surge', 'boom']
        negative_words = ['loss', 'decline', 'decrease', 'negative', 'failure', 'strike', 'disruption', 'crisis']
        
        pos_count = sum(1 for word in positive_words if word in full_text)
        neg_count = sum(1 for word in negative_words if word in full_text)
        
        if pos_count > neg_count:
            return 'positive'
        elif neg_count > pos_count:
            return 'negative'
        else:
            return 'neutral'
    
    # Use VADER sentiment analysis (same as training code)
    full_text = f"{heading} {text}"
    
    # Check if text is empty or NaN
    if pd.isna(full_text) or full_text.strip() == '':
        return 'neutral'
    
    try:
        # Get VADER sentiment scores
        score = vader_analyzer.polarity_scores(full_text)
        compound = score['compound']
        
        # Apply same thresholds as in training code
        if compound >= 0.05:
            return 'positive'
        elif compound <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    except Exception as e:
        logger.warning(f"VADER sentiment analysis failed: {e}, falling back to simple analysis")
        # Fallback to simple analysis
        full_text = f"{heading} {text}".lower()
        positive_words = ['profit', 'growth', 'increase', 'positive', 'success', 'record', 'surge', 'boom']
        negative_words = ['loss', 'decline', 'decrease', 'negative', 'failure', 'strike', 'disruption', 'crisis']
        
        pos_count = sum(1 for word in positive_words if word in full_text)
        neg_count = sum(1 for word in negative_words if word in full_text)
        
        if pos_count > neg_count:
            return 'positive'
        elif neg_count > pos_count:
            return 'negative'
        else:
            return 'neutral'

def create_advanced_features(news_input: NewsInput) -> Dict[str, Any]:
    """Create all advanced features for prediction"""
    # Temporal features
    temporal = create_temporal_features(news_input.date)
    
    # Text features
    text_features = create_text_features(news_input.heading, news_input.text)
    
    # News classification
    news_type, hotel_group = classify_news_type_and_hotel(news_input.heading, news_input.text)
    
    # Sentiment analysis
    sentiment = get_sentiment(news_input.heading, news_input.text)
    sentiment_encoded = get_sentiment_encoded(sentiment)
    
    # Source credibility
    source_credibility = get_source_credibility(news_input.source)
    
    # Encode categorical features
    type_encoded = {'hotel': 0, 'tourism': 1, 'general': 2}.get(news_type, 2)
    hotel_group_encoded = hotel_codes.index(hotel_group) if hotel_group in hotel_codes else 0
    
    # Interaction features
    sentiment_source_interaction = sentiment_encoded * source_credibility
    type_sentiment_interaction = type_encoded * sentiment_encoded
    
    # Combine all features
    features = {
        **temporal,
        **text_features,
        'sentiment_encoded': sentiment_encoded,
        'sentiment_abs': abs(sentiment_encoded),
        'source_credibility': source_credibility,
        'type_encoded': type_encoded,
        'hotel_group_encoded': hotel_group_encoded,
        'sentiment_source_interaction': sentiment_source_interaction,
        'type_sentiment_interaction': type_sentiment_interaction,
        'daily_news_count': 1,  # Default value
        'hotel_daily_news_count': 1,  # Default value
        'linked_to_search_spike': 0,  # Default value
        'spike_linked': 0,  # Default value
        'weight': 1.0,  # Default value
        'hotel_volatility': 0.5  # Default value
    }
    
    return features

def load_models():
    """Load all trained models"""
    global models, vectorizer, label_encoders, hotel_codes
    
    try:
        model_dir = "models/news/models"
        
        # Check if model directory exists
        if not os.path.exists(model_dir):
            logger.error(f"❌ Model directory not found: {model_dir}")
            logger.info("💡 Please ensure your trained models are in the correct location")
            return False
        
        # Load spike model with compatibility handling
        try:
            models['spike_model'] = joblib.load(f"{model_dir}/spike_model.pkl")
            logger.info("✅ Spike model loaded")
        except Exception as e:
            logger.error(f"❌ Error loading spike model: {str(e)}")
            logger.info("💡 This might be due to NumPy version compatibility")
            return False
        
        # Load vectorizer with compatibility handling
        try:
            vectorizer = joblib.load(f"{model_dir}/vectorizer.pkl")
            logger.info("✅ Vectorizer loaded")
        except Exception as e:
            logger.error(f"❌ Error loading vectorizer: {str(e)}")
            return False
        
        # Load label encoders with compatibility handling
        try:
            label_encoders = joblib.load(f"{model_dir}/label_encoders.pkl")
            logger.info("✅ Label encoders loaded")
        except Exception as e:
            logger.error(f"❌ Error loading label encoders: {str(e)}")
            return False
        
        # Load price models with compatibility handling
        price_models_dir = f"{model_dir}/Price_Models"
        models['price_models'] = {}
        
        if os.path.exists(price_models_dir):
            loaded_count = 0
            for filename in os.listdir(price_models_dir):
                if filename.startswith('price_model_') and filename.endswith('.pkl'):
                    hotel_code = filename[12:-4]  # Extract hotel code
                    model_path = os.path.join(price_models_dir, filename)
                    
                    try:
                        models['price_models'][hotel_code] = joblib.load(model_path)
                        hotel_codes.append(hotel_code)
                        loaded_count += 1
                    except Exception as e:
                        logger.warning(f"⚠️ Error loading price model for {hotel_code}: {str(e)}")
                        continue
            
            logger.info(f"✅ Loaded {loaded_count} price models")
            logger.info(f"✅ Hotel codes: {hotel_codes}")
            
            if loaded_count == 0:
                logger.error("❌ No price models could be loaded")
                return False
        else:
            logger.error(f"❌ Price models directory not found: {price_models_dir}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading models: {str(e)}")
        logger.info("💡 This might be due to:")
        logger.info("   1. NumPy version compatibility issues")
        logger.info("   2. Missing model files")
        logger.info("   3. Incorrect model directory structure")
        return False

def predict_single_article(news_input: NewsInput) -> PredictionResponse:
    """Predict for a single news article"""
    try:
        # Create features
        features = create_advanced_features(news_input)
        
        # Prepare text for vectorization
        text = f"{news_input.heading} {news_input.text}"
        
        # Spike prediction
        X_text_spike = vectorizer.transform([text])
        
        # Prepare numerical features for spike
        spike_feature_cols = [
            'sentiment_encoded', 'source_credibility', 'type_encoded',
            'hotel_group_encoded', 'day_of_week', 'month', 'quarter',
            'text_length', 'word_count', 'avg_word_length', 'sentiment_abs',
            'sentiment_source_interaction', 'type_sentiment_interaction',
            'daily_news_count', 'hotel_daily_news_count', 'text_complexity',
            'is_weekend', 'is_month_start', 'is_month_end', 'is_holiday_season', 
            'is_peak_season'
        ]
        
        X_num_spike = csr_matrix([[features[col] for col in spike_feature_cols]])
        X_combined_spike = hstack([X_text_spike, X_num_spike], format='csr')
        
        # Predict spike
        spike_proba = models['spike_model'].predict_proba(X_combined_spike)[0, 1]
        spike_prediction = 'SPIKE' if spike_proba > 0.5 else 'NO_SPIKE'
        
        # Hotel price predictions
        hotel_predictions = {}
        
        for hotel_code in hotel_codes:
            try:
                # Prepare features for price prediction
                enhanced_features = spike_feature_cols + [
                    'text_length', 'word_count', 'avg_word_length', 'sentiment_abs',
                    'source_credibility', 'sentiment_source_interaction', 'type_sentiment_interaction',
                    'daily_news_count', 'hotel_daily_news_count', 'text_complexity',
                    'is_weekend', 'is_month_start', 'is_month_end', 'weight',
                    'is_holiday_season', 'is_peak_season', 'spike_linked', 'hotel_volatility'
                ]
                
                X_text_hotel = vectorizer.transform([text])
                X_num_hotel = csr_matrix([[features[col] for col in enhanced_features]])
                X_combined_hotel = hstack([X_text_hotel, X_num_hotel], format='csr')
                
                # Predict price movement
                price_proba = models['price_models'][hotel_code].predict_proba(X_combined_hotel)[0]
                price_pred_idx = np.argmax(price_proba)
                
                # Decode prediction
                le = label_encoders[hotel_code]
                price_prediction = le.inverse_transform([price_pred_idx])[0]
                price_confidence = price_proba[price_pred_idx]
                
                hotel_predictions[hotel_code] = {
                    'prediction': price_prediction,
                    'confidence': float(price_confidence),
                    'probabilities': {
                        'up': float(price_proba[0]) if len(price_proba) > 0 else 0,
                        'down': float(price_proba[1]) if len(price_proba) > 1 else 0,
                        'neutral': float(price_proba[2]) if len(price_proba) > 2 else 0
                    }
                }
                
            except Exception as e:
                logger.warning(f"Error predicting for {hotel_code}: {str(e)}")
                hotel_predictions[hotel_code] = {
                    'prediction': 'neutral',
                    'confidence': 0.0,
                    'probabilities': {'up': 0.0, 'down': 0.0, 'neutral': 1.0}
                }
        
        return PredictionResponse(
            article_id=f"article_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            date=news_input.date,
            heading=news_input.heading,
            source=news_input.source,
            spike_prediction=spike_prediction,
            spike_confidence=float(spike_proba),
            hotel_predictions=hotel_predictions
        )
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# API Endpoints

@router.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("🚀 Loading news prediction models...")
    
    # Try to load models with retry mechanism
    max_retries = 3
    for attempt in range(max_retries):
        try:
            success = load_models()
            if success:
                logger.info("🎉 All models loaded successfully!")
                return
            else:
                logger.warning(f"⚠️ Attempt {attempt + 1}/{max_retries} failed to load models")
                if attempt < max_retries - 1:
                    logger.info("🔄 Retrying in 2 seconds...")
                    import asyncio
                    await asyncio.sleep(2)
        except Exception as e:
            logger.error(f"❌ Attempt {attempt + 1}/{max_retries} failed with exception: {str(e)}")
            if attempt < max_retries - 1:
                logger.info("🔄 Retrying in 2 seconds...")
                import asyncio
                await asyncio.sleep(2)
    
    logger.error("❌ Failed to load models after all attempts")
    logger.info("💡 The API will still start but prediction endpoints will return errors")
    logger.info("💡 Please check:")
    logger.info("   1. Model files are in models/news/models/")
    logger.info("   2. NumPy version compatibility")
    logger.info("   3. All required dependencies are installed")

@router.get("/")
async def root():
    """Root endpoint for news prediction"""
    return {
        "message": "Hotel Stock Price Prediction API",
        "version": "1.0.0",
        "status": "running",
        "models_loaded": len(models) > 0
    }

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    models_loaded = len(models) > 0 and vectorizer is not None and label_encoders is not None
    total_models = len(models.get('price_models', {})) + (1 if 'spike_model' in models else 0)
    
    # Check model directory
    model_dir = "models/news/models"
    model_dir_exists = os.path.exists(model_dir)
    
    # Check specific model files
    model_files_status = {}
    if model_dir_exists:
        required_files = ["spike_model.pkl", "vectorizer.pkl", "label_encoders.pkl"]
        for file in required_files:
            file_path = os.path.join(model_dir, file)
            model_files_status[file] = {
                "exists": os.path.exists(file_path),
                "size_mb": round(os.path.getsize(file_path) / (1024 * 1024), 2) if os.path.exists(file_path) else 0
            }
        
        # Check price models directory
        price_models_dir = os.path.join(model_dir, "Price_Models")
        if os.path.exists(price_models_dir):
            price_models = [f for f in os.listdir(price_models_dir) if f.startswith('price_model_') and f.endswith('.pkl')]
            model_files_status["Price_Models"] = {
                "exists": True,
                "count": len(price_models),
                "models": price_models[:5]  # Show first 5
            }
        else:
            model_files_status["Price_Models"] = {
                "exists": False,
                "count": 0,
                "models": []
            }
    
    return {
        "status": "healthy" if models_loaded else "unhealthy",
        "models_loaded": models_loaded,
        "total_models": total_models,
        "model_directory": {
            "exists": model_dir_exists,
            "path": model_dir
        },
        "model_files": model_files_status,
        "hotel_codes_loaded": len(hotel_codes),
        "hotel_codes": hotel_codes[:5] if hotel_codes else [],  # Show first 5
        "uptime": "running",
        "recommendations": get_health_recommendations(models_loaded, model_dir_exists, model_files_status)
    }

def get_health_recommendations(models_loaded, model_dir_exists, model_files_status):
    """Get recommendations based on health check results"""
    recommendations = []
    
    if not model_dir_exists:
        recommendations.append("Create models/news/models/ directory")
        recommendations.append("Copy your trained models to the directory")
        recommendations.append("Run fix_numpy_compatibility.py to create mock models for testing")
    
    if model_dir_exists and not models_loaded:
        recommendations.append("Models exist but failed to load - check NumPy compatibility")
        recommendations.append("Run fix_numpy_compatibility.py to fix compatibility issues")
        recommendations.append("Check if all required model files are present")
    
    if models_loaded:
        recommendations.append("All models loaded successfully - API is ready for predictions")
    
    return recommendations

@router.get("/models/info")
async def get_model_info():
    """Get information about loaded models"""
    return {
        "model_status": "loaded" if len(models) > 0 else "not_loaded",
        "loaded_models": list(models.keys()),
        "hotel_codes": hotel_codes,
        "last_updated": datetime.now().isoformat()
    }

@router.post("/predict", response_model=PredictionResponse)
async def predict_single(news_input: NewsInput):
    """Predict for a single news article"""
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return predict_single_article(news_input)

@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(news_batch: NewsBatchInput):
    """Predict for multiple news articles"""
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    predictions = []
    for i, article in enumerate(news_batch.articles):
        try:
            prediction = predict_single_article(article)
            prediction.article_id = f"article_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            predictions.append(prediction)
        except Exception as e:
            logger.error(f"Error predicting article {i+1}: {str(e)}")
            continue
    
    # Create summary
    total_articles = len(news_batch.articles)
    successful_predictions = len(predictions)
    spike_count = sum(1 for p in predictions if p.spike_prediction == 'SPIKE')
    
    summary = {
        "total_articles": total_articles,
        "successful_predictions": successful_predictions,
        "failed_predictions": total_articles - successful_predictions,
        "spike_predictions": spike_count,
        "no_spike_predictions": successful_predictions - spike_count,
        "average_spike_confidence": np.mean([p.spike_confidence for p in predictions]) if predictions else 0
    }
    
    return BatchPredictionResponse(predictions=predictions, summary=summary)

@router.post("/predict/upload")
async def predict_from_file(file: UploadFile = File(...)):
    """Predict from uploaded CSV file"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        # Read CSV
        df = pd.read_csv(file.file)
        
        # Validate required columns
        required_cols = ['date', 'heading', 'text', 'source']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Missing required columns: {missing_cols}")
        
        # Convert to NewsInput objects
        articles = []
        for _, row in df.iterrows():
            articles.append(NewsInput(
                date=str(row['date']),
                heading=str(row['heading']),
                text=str(row['text']),
                source=str(row['source'])
            ))
        
        # Make predictions
        news_batch = NewsBatchInput(articles=articles)
        
        # Process predictions
        predictions = []
        for i, article in enumerate(articles):
            try:
                prediction = predict_single_article(article)
                prediction.article_id = f"article_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Error predicting article {i+1}: {str(e)}")
                continue
        
        # Save results to CSV
        output_data = []
        for pred in predictions:
            row = {
                'article_id': pred.article_id,
                'date': pred.date,
                'heading': pred.heading,
                'source': pred.source,
                'spike_prediction': pred.spike_prediction,
                'spike_confidence': pred.spike_confidence
            }
            
            # Add hotel predictions
            for hotel_code, hotel_pred in pred.hotel_predictions.items():
                row[f'{hotel_code}_prediction'] = hotel_pred['prediction']
                row[f'{hotel_code}_confidence'] = hotel_pred['confidence']
                row[f'{hotel_code}_up_prob'] = hotel_pred['probabilities']['up']
                row[f'{hotel_code}_down_prob'] = hotel_pred['probabilities']['down']
                row[f'{hotel_code}_neutral_prob'] = hotel_pred['probabilities']['neutral']
            
            output_data.append(row)
        
        # Save to temporary file
        output_df = pd.DataFrame(output_data)
        output_path = f"temp_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        output_df.to_csv(output_path, index=False)
        
        return FileResponse(
            path=output_path,
            filename=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            media_type='text/csv'
        )
        
    except Exception as e:
        logger.error(f"Error processing uploaded file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@router.get("/hotels")
async def get_hotels():
    """Get list of supported hotels"""
    return {
        "hotels": hotel_codes,
        "count": len(hotel_codes),
        "description": "List of hotel codes for which price predictions are available"
    }

@router.get("/features")
async def get_feature_info():
    """Get information about features used in the model"""
    return {
        "temporal_features": [
            "day_of_week", "month", "quarter", "year", "is_weekend",
            "is_month_start", "is_month_end", "is_holiday_season", "is_peak_season"
        ],
        "text_features": [
            "text_length", "word_count", "avg_word_length", "text_complexity"
        ],
        "sentiment_features": [
            "sentiment_encoded", "sentiment_abs", "sentiment_source_interaction"
        ],
        "source_features": [
            "source_credibility", "type_encoded", "hotel_group_encoded"
        ],
        "interaction_features": [
            "sentiment_source_interaction", "type_sentiment_interaction"
        ],
        "volume_features": [
            "daily_news_count", "hotel_daily_news_count"
        ]
    }

@router.post("/models/reload")
async def reload_models():
    """Reload all models"""
    try:
        global models, vectorizer, label_encoders, hotel_codes
        
        # Clear existing models
        models = {}
        vectorizer = None
        label_encoders = None
        hotel_codes = []
        
        logger.info("🔄 Manually reloading models...")
        success = load_models()
        if success:
            return {"message": "Models reloaded successfully", "status": "success"}
        else:
            raise HTTPException(status_code=500, detail="Failed to reload models")
    except Exception as e:
        logger.error(f"Error reloading models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error reloading models: {str(e)}")

@router.post("/models/debug")
async def debug_models():
    """Debug model loading issues"""
    try:
        import traceback
        model_dir = "models/news/models"
        
        debug_info = {
            "model_directory": model_dir,
            "directory_exists": os.path.exists(model_dir),
            "numpy_version": np.__version__,
            "joblib_version": joblib.__version__,
            "sklearn_version": None,
            "errors": []
        }
        
        # Try to get sklearn version
        try:
            import sklearn
            debug_info["sklearn_version"] = sklearn.__version__
        except:
            debug_info["sklearn_version"] = "Not available"
        
        # Try loading each model individually
        model_files = ["spike_model.pkl", "vectorizer.pkl", "label_encoders.pkl"]
        
        for model_file in model_files:
            model_path = os.path.join(model_dir, model_file)
            if os.path.exists(model_path):
                try:
                    logger.info(f"🔄 Testing load of {model_file}")
                    test_model = joblib.load(model_path)
                    debug_info[f"{model_file}_loaded"] = True
                    debug_info[f"{model_file}_type"] = str(type(test_model))
                except Exception as e:
                    debug_info[f"{model_file}_loaded"] = False
                    debug_info[f"{model_file}_error"] = str(e)
                    debug_info["errors"].append(f"{model_file}: {str(e)}")
                    logger.error(f"❌ Error loading {model_file}: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
            else:
                debug_info[f"{model_file}_loaded"] = False
                debug_info[f"{model_file}_error"] = "File not found"
        
        # Try loading price models
        price_models_dir = os.path.join(model_dir, "Price_Models")
        if os.path.exists(price_models_dir):
            price_models = [f for f in os.listdir(price_models_dir) if f.startswith('price_model_') and f.endswith('.pkl')]
            debug_info["price_models_count"] = len(price_models)
            
            # Test first few price models
            for i, model_file in enumerate(price_models[:3]):
                model_path = os.path.join(price_models_dir, model_file)
                try:
                    logger.info(f"🔄 Testing load of {model_file}")
                    test_model = joblib.load(model_path)
                    debug_info[f"price_model_{i}_loaded"] = True
                except Exception as e:
                    debug_info[f"price_model_{i}_loaded"] = False
                    debug_info[f"price_model_{i}_error"] = str(e)
                    debug_info["errors"].append(f"{model_file}: {str(e)}")
        
        return debug_info
        
    except Exception as e:
        logger.error(f"Error in debug: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in debug: {str(e)}") 

@router.post("/test/sentiment")
async def test_sentiment_analysis(text: str):
    """Test sentiment analysis with sample text"""
    try:
        # Test VADER sentiment
        if VADER_AVAILABLE:
            score = vader_analyzer.polarity_scores(text)
            vader_sentiment = get_sentiment("", text)
            return {
                "text": text,
                "vader_scores": score,
                "vader_sentiment": vader_sentiment,
                "vader_available": True
            }
        else:
            simple_sentiment = get_sentiment("", text)
            return {
                "text": text,
                "simple_sentiment": simple_sentiment,
                "vader_available": False,
                "message": "VADER not available, using simple keyword-based analysis"
            }
    except Exception as e:
        logger.error(f"Error in sentiment test: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in sentiment test: {str(e)}")

@router.post("/test/features")
async def test_feature_engineering(news_input: NewsInput):
    """Test feature engineering with sample news article"""
    try:
        # Create features
        features = create_advanced_features(news_input)
        
        # Get sentiment
        sentiment = get_sentiment(news_input.heading, news_input.text)
        
        # Get news classification
        news_type, hotel_group = classify_news_type_and_hotel(news_input.heading, news_input.text)
        
        return {
            "input": {
                "date": news_input.date,
                "heading": news_input.heading,
                "text": news_input.text[:100] + "..." if len(news_input.text) > 100 else news_input.text,
                "source": news_input.source
            },
            "sentiment": sentiment,
            "news_type": news_type,
            "hotel_group": hotel_group,
            "features": features,
            "vader_available": VADER_AVAILABLE
        }
    except Exception as e:
        logger.error(f"Error in feature test: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in feature test: {str(e)}") 