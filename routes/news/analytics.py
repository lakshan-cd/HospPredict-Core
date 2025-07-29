from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/news/analytics", tags=["news-analytics"])

# Pydantic models
class PerformanceMetrics(BaseModel):
    hotel_code: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    total_predictions: int

class PredictionTrend(BaseModel):
    date: str
    spike_count: int
    total_predictions: int
    spike_rate: float
    avg_confidence: float

class HotelInsights(BaseModel):
    hotel_code: str
    most_common_prediction: str
    prediction_distribution: Dict[str, int]
    avg_confidence: float
    volatility_score: float

# In-memory storage for analytics (in production, use a database)
prediction_history = []
performance_metrics = {}

@router.get("/performance", response_model=List[PerformanceMetrics])
async def get_model_performance():
    """Get performance metrics for all hotel models"""
    try:
        # This would typically fetch from a database
        # For now, return mock data
        return [
            PerformanceMetrics(
                hotel_code="AHUN.N0000",
                accuracy=0.75,
                precision=0.72,
                recall=0.68,
                f1_score=0.70,
                auc_score=0.78,
                total_predictions=1000
            ),
            PerformanceMetrics(
                hotel_code="TRAN.N0000",
                accuracy=0.73,
                precision=0.71,
                recall=0.65,
                f1_score=0.68,
                auc_score=0.76,
                total_predictions=950
            )
        ]
    except Exception as e:
        logger.error(f"Error fetching performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching performance metrics")

@router.get("/trends", response_model=List[PredictionTrend])
async def get_prediction_trends(days: int = 30):
    """Get prediction trends over time"""
    try:
        # Generate mock trend data
        trends = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        current_date = start_date
        while current_date <= end_date:
            # Mock data - in production, aggregate from prediction_history
            spike_count = np.random.randint(5, 25)
            total_predictions = np.random.randint(50, 100)
            spike_rate = spike_count / total_predictions if total_predictions > 0 else 0
            avg_confidence = np.random.uniform(0.6, 0.9)
            
            trends.append(PredictionTrend(
                date=current_date.strftime("%Y-%m-%d"),
                spike_count=spike_count,
                total_predictions=total_predictions,
                spike_rate=spike_rate,
                avg_confidence=avg_confidence
            ))
            
            current_date += timedelta(days=1)
        
        return trends
    except Exception as e:
        logger.error(f"Error fetching prediction trends: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching prediction trends")

@router.get("/hotels/insights", response_model=List[HotelInsights])
async def get_hotel_insights():
    """Get insights for each hotel"""
    try:
        hotel_codes = [
            "AHUN.N0000", "TRAN.N0000", "STAF.N0000", "MRH.N0000", "AHPL.N0000",
            "GHLL.N0000", "KJL.N0000", "EDEN.N0000", "RENU.N0000", "RPBH.N0000"
        ]
        
        insights = []
        for hotel_code in hotel_codes:
            # Mock insights - in production, analyze prediction_history
            predictions = ["up", "down", "neutral"]
            most_common = np.random.choice(predictions, p=[0.4, 0.3, 0.3])
            
            distribution = {
                "up": np.random.randint(30, 50),
                "down": np.random.randint(20, 40),
                "neutral": np.random.randint(25, 45)
            }
            
            insights.append(HotelInsights(
                hotel_code=hotel_code,
                most_common_prediction=most_common,
                prediction_distribution=distribution,
                avg_confidence=np.random.uniform(0.65, 0.85),
                volatility_score=np.random.uniform(0.2, 0.8)
            ))
        
        return insights
    except Exception as e:
        logger.error(f"Error fetching hotel insights: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching hotel insights")

@router.get("/sentiment/analysis")
async def get_sentiment_analysis(days: int = 7):
    """Get sentiment analysis trends"""
    try:
        # Mock sentiment analysis data
        sentiments = ["positive", "negative", "neutral"]
        sentiment_counts = {
            "positive": np.random.randint(20, 40),
            "negative": np.random.randint(10, 25),
            "neutral": np.random.randint(15, 30)
        }
        
        total = sum(sentiment_counts.values())
        sentiment_distribution = {k: v/total for k, v in sentiment_counts.items()}
        
        return {
            "period_days": days,
            "total_articles": total,
            "sentiment_distribution": sentiment_distribution,
            "sentiment_counts": sentiment_counts,
            "dominant_sentiment": max(sentiment_counts, key=sentiment_counts.get)
        }
    except Exception as e:
        logger.error(f"Error fetching sentiment analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching sentiment analysis")

@router.get("/sources/credibility")
async def get_source_credibility_analysis():
    """Get analysis of news source credibility and impact"""
    try:
        sources = ["economynext", "dailynews", "sundaytimes", "island", "colombopage"]
        
        source_analysis = {}
        for source in sources:
            source_analysis[source] = {
                "credibility_score": np.random.uniform(0.6, 0.9),
                "article_count": np.random.randint(50, 200),
                "avg_spike_rate": np.random.uniform(0.1, 0.4),
                "avg_confidence": np.random.uniform(0.6, 0.85)
            }
        
        return {
            "source_analysis": source_analysis,
            "most_credible_source": max(source_analysis.keys(), 
                                      key=lambda x: source_analysis[x]["credibility_score"]),
            "total_sources": len(sources)
        }
    except Exception as e:
        logger.error(f"Error fetching source credibility analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching source credibility analysis")

@router.get("/predictions/summary")
async def get_predictions_summary():
    """Get summary of recent predictions"""
    try:
        # Mock summary data
        return {
            "total_predictions_today": np.random.randint(100, 500),
            "spike_predictions_today": np.random.randint(10, 50),
            "avg_confidence_today": np.random.uniform(0.65, 0.85),
            "most_predicted_hotel": "AHUN.N0000",
            "most_common_prediction": "neutral",
            "prediction_accuracy_trend": "improving"
        }
    except Exception as e:
        logger.error(f"Error fetching predictions summary: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching predictions summary")

@router.post("/predictions/record")
async def record_prediction(prediction_data: Dict[str, Any]):
    """Record a prediction for analytics (called internally)"""
    try:
        prediction_data["timestamp"] = datetime.now().isoformat()
        prediction_history.append(prediction_data)
        
        # Keep only last 10000 predictions to prevent memory issues
        if len(prediction_history) > 10000:
            prediction_history.pop(0)
        
        return {"message": "Prediction recorded successfully"}
    except Exception as e:
        logger.error(f"Error recording prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Error recording prediction") 