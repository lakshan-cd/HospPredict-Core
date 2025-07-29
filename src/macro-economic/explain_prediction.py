#!/usr/bin/env python3
"""
Explain what price the macro-economic API is predicting
"""

import os
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Paths
SEQUENCE_DIR = "data/macro-economic/data/processed/sequences"
MODEL_DIR = "models/macro-economic/models"
OUTPUT_DIR = "data/macro-economic/outputs"

def explain_prediction(company_name: str):
    """Explain what price is being predicted for a company"""
    
    print(f"🔍 Explaining prediction for: {company_name}")
    print("=" * 60)
    
    # Load performance data to see the timeline
    performance_path = os.path.join(OUTPUT_DIR, f"{company_name}_forecast_vs_actual.csv")
    
    if not os.path.exists(performance_path):
        print(f"❌ Performance data not found: {performance_path}")
        return
    
    df = pd.read_csv(performance_path)
    
    print(f"📊 Performance data shape: {df.shape}")
    print(f"📅 Data points: {len(df)} days")
    
    # Show last few days
    print(f"\n📈 Last 5 days of data:")
    print(df.tail().to_string(index=False))
    
    # Get current and predicted prices
    current_price = df['Actual Price'].iloc[-1]
    predicted_price = df['Predicted Price'].iloc[-1]
    
    print(f"\n💰 Price Analysis:")
    print(f"   Current Price (Last Known Close): {current_price:.2f} Rs.")
    print(f"   Predicted Price (Next Day Close): {predicted_price:.2f} Rs.")
    print(f"   Price Change: {predicted_price - current_price:.2f} Rs.")
    print(f"   Percentage Change: {((predicted_price - current_price) / current_price) * 100:.2f}%")
    
    # Load sequences to show what data is used for prediction
    try:
        X_stock = np.load(os.path.join(SEQUENCE_DIR, f"{company_name}_X_stock.npy"))
        X_macro = np.load(os.path.join(SEQUENCE_DIR, f"{company_name}_X_macro.npy"))
        
        print(f"\n🔢 Sequence Data Used for Prediction:")
        print(f"   Stock sequence shape: {X_stock.shape}")
        print(f"   Macro sequence shape: {X_macro.shape}")
        print(f"   Window size: {X_stock.shape[1]} days")
        
        # Show the last sequence used for prediction
        last_stock_seq = X_stock[-1]  # Last 30 days of closing prices
        print(f"\n📊 Last 30 days of closing prices used for prediction:")
        for i, price in enumerate(last_stock_seq.flatten()):
            print(f"   Day {i+1}: {price:.4f}")
            
    except Exception as e:
        print(f"❌ Error loading sequences: {str(e)}")
    
    print(f"\n🎯 Summary:")
    print(f"   The API predicts the NEXT TRADING DAY's CLOSING PRICE")
    print(f"   It uses the last 30 days of closing prices and macro indicators")
    print(f"   Current price = Last known closing price from training data")
    print(f"   Predicted price = Expected closing price for the next trading day")

def show_all_companies():
    """Show prediction explanation for all companies"""
    
    print("🏢 Showing prediction explanation for all companies")
    print("=" * 60)
    
    # Get all performance files
    performance_files = []
    if os.path.exists(OUTPUT_DIR):
        for fname in os.listdir(OUTPUT_DIR):
            if fname.endswith("_forecast_vs_actual.csv"):
                company = fname.replace("_forecast_vs_actual.csv", "")
                performance_files.append(company)
    
    print(f"Found {len(performance_files)} companies")
    
    for company in sorted(performance_files):
        try:
            explain_prediction(company)
            print("\n" + "-" * 60 + "\n")
        except Exception as e:
            print(f"❌ Error explaining {company}: {str(e)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        company_name = sys.argv[1]
        explain_prediction(company_name)
    else:
        show_all_companies() 