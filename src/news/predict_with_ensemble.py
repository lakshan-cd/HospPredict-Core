#!/usr/bin/env python3
"""
Prediction Script for Enhanced Ensemble Model
Make predictions for new news articles using the trained ensemble model
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.news.enhanced_ensemble_model import EnhancedEnsembleModel

class NewsPredictor:
    """
    Class for making predictions with the trained ensemble model
    """
    
    def __init__(self, models_path="models/news/enhanced_ensemble/"):
        self.models_path = models_path
        self.ensemble_model = EnhancedEnsembleModel()
        self.load_models()
        
    def load_models(self):
        """Load the trained models"""
        try:
            self.ensemble_model.load_models(self.models_path)
            print("✅ Models loaded successfully")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
    
    def preprocess_news_article(self, heading, content, source, date=None):
        """
        Preprocess a single news article for prediction
        """
        if date is None:
            date = datetime.now()
        
        # Create a single-row DataFrame
        df = pd.DataFrame([{
            'heading': heading,
            'content': content,
            'source': source,
            'date': date
        }])
        
        # Apply the same preprocessing as in training
        df = self.ensemble_model.create_advanced_features(df)
        
        # Extract features for prediction
        feature_cols = [
            'sentiment_encoded', 'source_credibility', 'type_encoded', 
            'hotel_group_encoded', 'day_of_week', 'month', 'quarter',
            'text_length', 'word_count', 'avg_word_length', 'sentiment_abs',
            'sentiment_source_interaction', 'type_sentiment_interaction',
            'daily_news_count', 'hotel_daily_news_count', 'text_complexity',
            'is_weekend', 'is_month_start', 'is_month_end'
        ]
        
        # Fill missing values
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
            else:
                df[col] = df[col].fillna(0)
        
        return df[feature_cols].values[0]
    
    def predict_single_article(self, heading, content, source, date=None):
        """
        Make predictions for a single news article
        """
        print("🔮 Making predictions...")
        
        # Preprocess the article
        features = self.preprocess_news_article(heading, content, source, date)
        
        # Combine heading and content
        text = f"{heading} {content}"
        
        # Make prediction
        prediction = self.ensemble_model.predict(text, features)
        
        return prediction
    
    def predict_batch(self, articles_df):
        """
        Make predictions for a batch of news articles
        """
        print(f"🔮 Making predictions for {len(articles_df)} articles...")
        
        predictions = []
        
        for idx, row in articles_df.iterrows():
            try:
                prediction = self.predict_single_article(
                    row['heading'], 
                    row['content'], 
                    row['source'], 
                    row.get('date')
                )
                
                # Add article info to prediction
                prediction['article_id'] = idx
                prediction['heading'] = row['heading'][:100] + "..." if len(row['heading']) > 100 else row['heading']
                
                predictions.append(prediction)
                
            except Exception as e:
                print(f"⚠️ Error predicting article {idx}: {e}")
                continue
        
        return predictions
    
    def format_predictions(self, predictions):
        """
        Format predictions into a readable format
        """
        formatted_results = []
        
        for pred in predictions:
            result = {
                'Article ID': pred.get('article_id', 'N/A'),
                'Heading': pred.get('heading', 'N/A'),
                'Spike Probability': f"{pred['spike_probability']:.3f}",
                'Ensemble Probability': f"{pred['ensemble_probability']:.3f}",
                'Spike Prediction': 'YES' if pred['spike_probability'] > 0.5 else 'NO',
                'Ensemble Prediction': 'YES' if pred['ensemble_probability'] > 0.5 else 'NO'
            }
            
            # Add price predictions for each hotel
            for hotel, price_pred in pred['price_predictions'].items():
                result[f'{hotel}_Direction'] = price_pred['prediction']
                result[f'{hotel}_Confidence'] = f"{max(price_pred['probabilities']):.3f}"
            
            formatted_results.append(result)
        
        return pd.DataFrame(formatted_results)
    
    def get_top_hotels(self, prediction, top_k=5):
        """
        Get top k hotels with highest confidence in price movement
        """
        hotel_confidences = []
        
        for hotel, price_pred in prediction['price_predictions'].items():
            confidence = max(price_pred['probabilities'])
            direction = price_pred['prediction']
            hotel_confidences.append({
                'hotel': hotel,
                'direction': direction,
                'confidence': confidence
            })
        
        # Sort by confidence
        hotel_confidences.sort(key=lambda x: x['confidence'], reverse=True)
        
        return hotel_confidences[:top_k]

def main():
    """
    Example usage of the prediction system
    """
    print("🎯 Enhanced Ensemble Prediction System")
    print("=" * 50)
    
    # Initialize predictor
    try:
        predictor = NewsPredictor()
    except Exception as e:
        print(f"❌ Failed to initialize predictor: {e}")
        return
    
    # Example 1: Single article prediction
    print("\n📰 Example 1: Single Article Prediction")
    print("-" * 40)
    
    sample_article = {
        'heading': 'John Keells Hotels Reports Strong Q3 Performance',
        'content': 'John Keells Hotels PLC announced impressive third-quarter results with revenue growth of 15% year-over-year. The company attributed the strong performance to increased tourist arrivals and improved operational efficiency.',
        'source': 'economynext',
        'date': datetime.now()
    }
    
    prediction = predictor.predict_single_article(
        sample_article['heading'],
        sample_article['content'],
        sample_article['source'],
        sample_article['date']
    )
    
    print(f"📊 Spike Probability: {prediction['spike_probability']:.3f}")
    print(f"🎯 Ensemble Probability: {prediction['ensemble_probability']:.3f}")
    
    # Get top hotel predictions
    top_hotels = predictor.get_top_hotels(prediction, top_k=3)
    print("\n🏨 Top Hotel Predictions:")
    for hotel_pred in top_hotels:
        print(f"  {hotel_pred['hotel']}: {hotel_pred['direction']} (confidence: {hotel_pred['confidence']:.3f})")
    
    # Example 2: Batch prediction from CSV
    print("\n📰 Example 2: Batch Prediction")
    print("-" * 40)
    
    # Create sample batch data
    sample_batch = pd.DataFrame([
        {
            'heading': 'Sri Lanka Tourism Shows Recovery Signs',
            'content': 'Tourism arrivals to Sri Lanka increased by 25% in the latest quarter, signaling a strong recovery in the hospitality sector.',
            'source': 'dailynews',
            'date': datetime.now()
        },
        {
            'heading': 'Aitken Spence Announces New Hotel Development',
            'content': 'Aitken Spence Hotels PLC has announced plans to develop a new luxury resort in the southern coast of Sri Lanka.',
            'source': 'sundaytimes',
            'date': datetime.now()
        }
    ])
    
    batch_predictions = predictor.predict_batch(sample_batch)
    formatted_results = predictor.format_predictions(batch_predictions)
    
    print("\n📊 Batch Prediction Results:")
    print(formatted_results.to_string(index=False))
    
    # Save results
    output_path = "predictions/enhanced_ensemble_predictions.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    formatted_results.to_csv(output_path, index=False)
    print(f"\n✅ Predictions saved to {output_path}")

if __name__ == "__main__":
    main() 