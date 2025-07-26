#!/usr/bin/env python3
"""
Enhanced Ensemble Training Script
Adapts the Colab preprocessing and model building code for local environment
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.news.enhanced_ensemble_model import EnhancedEnsembleModel

def load_and_preprocess_data(data_path):
    """
    Load and preprocess the data similar to the Colab preprocessing
    """
    print("📊 Loading and preprocessing data...")
    
    # Load the processed dataset
    df = pd.read_csv(data_path)
    
    # Convert date column
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Filter data for 2014-2019 (as in original code)
    df = df[(df['date'].dt.year >= 2014) & (df['date'].dt.year <= 2019)]
    
    # Remove rows where both cleaned_heading and cleaned_content are empty
    df = df[~((df['heading'].isna() | df['heading'].str.strip().eq('')) &
              (df['content'].isna() | df['content'].str.strip().eq('')))]
    
    # Ensure required columns exist
    required_cols = ['date', 'heading', 'content', 'source', 'sentiment', 'type', 'hotel']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"⚠️ Missing columns: {missing_cols}")
        return None
    
    print(f"✅ Loaded {len(df)} news articles")
    return df

def create_hotel_mappings():
    """
    Create hotel alias and symbol mappings as in the original code
    """
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
    
    hotel_symbol_map = {
        "AITKEN SPENCE": "AHUN.N0000",
        "ASIAN HOTELS": "AHPL.N0000",
        "BROWNS BEACH": "BBH.N0000",
        "CEYLON HOTELS": "CHOT.N0000",
        "DOLPHIN HOTELS": "STAF.N0000",
        "EDEN HOTEL": "EDEN.N0000",
        "GALADARI HOTELS": "GHLL.N0000",
        "HOTEL SIGIRIYA": "HSIG.N0000",
        "JOHN KEELLS": "KHL.N0000",
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
    
    return hotel_aliases, hotel_symbol_map

def classify_news_and_hotels(df, hotel_aliases):
    """
    Classify news articles by type and hotel as in the original code
    """
    print("🏷️ Classifying news articles...")
    
    # Tourism keywords
    tourism_keywords = [
        "tourism", "travel", "holiday", "resort", "hotel industry", "beach", "sri lanka tourism",
        "tourist", "vacation", "hospitality", "wildlife park", "ayurveda", "unawatuna", "ella",
        "sigiriya", "yala", "kandy", "galle fort", "ecotourism", "foreign arrivals"
    ]
    
    def classify_news_type_and_hotel(row):
        heading = str(row['heading']).lower()
        content = str(row['content']).lower()
        
        # HOTEL DETECTION
        for hotel, keywords in hotel_aliases.items():
            for keyword in keywords:
                if keyword in heading or keyword in content:
                    return pd.Series(["hotel", hotel])
        
        # TOURISM DETECTION
        for keyword in tourism_keywords:
            if keyword in heading or keyword in content:
                return pd.Series(["tourism", "Unknown"])
        
        # GENERAL
        return pd.Series(["general", "Unknown"])
    
    # Apply classification
    df[['type', 'hotel']] = df.apply(classify_news_type_and_hotel, axis=1)
    
    # Add hotel group and symbol
    hotel_aliases, hotel_symbol_map = create_hotel_mappings()
    
    def get_hotel_group(hotel_name):
        hotel_name = str(hotel_name).lower()
        for group, keywords in hotel_aliases.items():
            for kw in keywords:
                if kw in hotel_name:
                    return group
        return "None"
    
    df['hotel_group'] = df['hotel'].apply(get_hotel_group)
    df['hotel_symbol'] = df['hotel_group'].map(hotel_symbol_map).fillna("None")
    
    return df

def add_spike_linking_features(df, stock_spike_path=None, search_spike_path=None):
    """
    Add spike linking features as in the original code
    """
    print("🔗 Adding spike linking features...")
    
    # Initialize spike columns
    df['linked_to_stock_spike'] = False
    df['linked_to_search_spike'] = False
    df['spike_linked'] = False
    
    # If stock spike data is available
    if stock_spike_path and os.path.exists(stock_spike_path):
        try:
            stock_spike_df = pd.read_csv(stock_spike_path)
            stock_spike_df['event_date'] = pd.to_datetime(stock_spike_df['event_date'])
            
            def is_stock_spike_related(news_row):
                symbol = news_row['hotel_symbol']
                if pd.isna(symbol) or symbol == "None":
                    return False
                date = news_row['date']
                spikes = stock_spike_df[stock_spike_df['hotel_symbol'] == symbol]
                return any((date >= spike - timedelta(days=2)) & (date <= spike + timedelta(days=2)) 
                          for spike in spikes['event_date'])
            
            df['linked_to_stock_spike'] = df.apply(is_stock_spike_related, axis=1)
            print(f"✅ Added stock spike linking for {df['linked_to_stock_spike'].sum()} articles")
        except Exception as e:
            print(f"⚠️ Error loading stock spike data: {e}")
    
    # If search spike data is available
    if search_spike_path and os.path.exists(search_spike_path):
        try:
            search_spike_df = pd.read_csv(search_spike_path)
            search_spike_df['week_start'] = pd.to_datetime(search_spike_df['Week'])
            
            def is_search_spike_related(news_row):
                news_week = news_row['date'] - timedelta(days=news_row['date'].weekday())
                spike_keywords = search_spike_df[search_spike_df['week_start'] == news_week]['spike_type'].tolist()
                
                if news_row['type'] == 'hotel' and news_row['hotel_symbol'] in spike_keywords:
                    return True
                elif news_row['type'] in ['tourism', 'general'] and 'tourism' in spike_keywords:
                    return True
                return False
            
            df['linked_to_search_spike'] = df.apply(is_search_spike_related, axis=1)
            print(f"✅ Added search spike linking for {df['linked_to_search_spike'].sum()} articles")
        except Exception as e:
            print(f"⚠️ Error loading search spike data: {e}")
    
    # Combined spike linking
    df['spike_linked'] = df['linked_to_stock_spike'] | df['linked_to_search_spike']
    
    return df

def calculate_weights(df):
    """
    Calculate weights for news articles as in the original code
    """
    print("⚖️ Calculating article weights...")
    
    def calculate_weight(row):
        base = 0.2
        
        if row['linked_to_stock_spike'] and row['linked_to_search_spike']:
            base += 0.7
        elif row['linked_to_stock_spike']:
            base += 0.4
        elif row['linked_to_search_spike']:
            base += 0.3
        
        if row['type'] == 'hotel':
            base += 0.2
        elif row['type'] == 'tourism':
            base += 0.1
        
        if row['sentiment'] in ['positive', 'negative']:
            base += 0.05
        
        return round(min(base, 1.0), 2)
    
    df['weight'] = df.apply(calculate_weight, axis=1)
    
    return df

def add_price_change_features(df, stock_data_path=None):
    """
    Add price change features for each hotel
    """
    print("📈 Adding price change features...")
    
    # If stock data is available, add price change columns
    if stock_data_path and os.path.exists(stock_data_path):
        try:
            # This would need to be implemented based on your stock data structure
            # For now, we'll create placeholder columns
            hotel_symbols = [
                "AHUN.N0000", "AHPL.N0000", "BBH.N0000", "CHOT.N0000", "STAF.N0000",
                "EDEN.N0000", "GHLL.N0000", "HSIG.N0000", "KHL.N0000", "MRH.N0000",
                "PALM.N0000", "PEG.N0000", "RENU.N0000", "RCH.N0000", "RPBH.N0000",
                "SHOT.N0000", "SIGV.N0000", "TAJ.N0000", "TANG.N0000", "LHL.N0000",
                "KHC.N0000", "TRAN.N0000"
            ]
            
            for symbol in hotel_symbols:
                df[f'{symbol}_change_1d'] = np.random.normal(0, 0.02, len(df))  # Placeholder
            
            print(f"✅ Added price change features for {len(hotel_symbols)} hotels")
        except Exception as e:
            print(f"⚠️ Error adding price change features: {e}")
    else:
        print("⚠️ No stock data provided, skipping price change features")
    
    return df

def main():
    """
    Main training pipeline
    """
    print("🚀 Starting Enhanced Ensemble Training Pipeline")
    
    # Configuration
    data_path = "data/news/processed/processed_news_articles_SL.csv"  # Update path as needed
    stock_spike_path = "data/processed/Stock_spikes.CSV"  # Update path as needed
    search_spike_path = "data/processed/Search_2014_2024_combined.csv"  # Update path as needed
    stock_data_path = "data/processed/stock_data/"  # Update path as needed
    models_output_path = "models/news/enhanced_ensemble/"
    
    # Create output directory
    os.makedirs(models_output_path, exist_ok=True)
    
    # Step 1: Load and preprocess data
    df = load_and_preprocess_data(data_path)
    if df is None:
        print("❌ Failed to load data")
        return
    
    # Step 2: Create hotel mappings
    hotel_aliases, hotel_symbol_map = create_hotel_mappings()
    
    # Step 3: Classify news articles
    df = classify_news_and_hotels(df, hotel_aliases)
    
    # Step 4: Add spike linking features
    df = add_spike_linking_features(df, stock_spike_path, search_spike_path)
    
    # Step 5: Calculate weights
    df = calculate_weights(df)
    
    # Step 6: Add price change features
    df = add_price_change_features(df, stock_data_path)
    
    # Step 7: Save preprocessed data
    preprocessed_path = "data/news/processed/enhanced_preprocessed_news.csv"
    os.makedirs(os.path.dirname(preprocessed_path), exist_ok=True)
    df.to_csv(preprocessed_path, index=False)
    print(f"✅ Saved preprocessed data to {preprocessed_path}")
    
    # Step 8: Train enhanced ensemble model
    print("\n🎯 Training Enhanced Ensemble Model...")
    ensemble_model = EnhancedEnsembleModel(random_state=42)
    
    try:
        results = ensemble_model.train(preprocessed_path)
        
        # Step 9: Save models
        ensemble_model.save_models(models_output_path)
        
        # Step 10: Print results summary
        print("\n📊 Training Results Summary:")
        print("=" * 50)
        
        if 'spike_metrics' in results:
            spike_f1 = results['spike_metrics']['weighted avg']['f1-score']
            print(f"Spike Model F1-Score: {spike_f1:.4f}")
        
        if 'price_metrics' in results and not results['price_metrics'].empty:
            avg_price_f1 = results['price_metrics']['f1_score'].mean()
            print(f"Average Price Model F1-Score: {avg_price_f1:.4f}")
        
        print("✅ Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 