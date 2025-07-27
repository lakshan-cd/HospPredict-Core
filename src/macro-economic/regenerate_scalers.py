#!/usr/bin/env python3
"""
Regenerate scalers for macro-economic models
This script recreates the scalers in the current environment to fix compatibility issues
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

# Constants
PROCESSED_DIR = "data/macro-economic/data/processed"
MODEL_DIR = "models/macro-economic/models"
MACRO_FEATURES = ['Tourism Arrivals', 'Money Supply', 'Exchange Rate', 'Inflation rate']

def regenerate_scalers():
    """Regenerate all scalers for compatibility"""
    
    print("Regenerating scalers for all companies...")
    
    # Get all processed data files
    processed_files = []
    if os.path.exists(PROCESSED_DIR):
        for fname in os.listdir(PROCESSED_DIR):
            if fname.endswith("_scaled.csv"):
                processed_files.append(fname)
    
    print(f"Found {len(processed_files)} processed data files")
    
    # Regenerate scalers for each company
    for filename in processed_files:
        try:
            company_name = filename.replace("_scaled.csv", "")
            print(f"\nProcessing scalers for: {company_name}")
            
            # Load processed data
            data_path = os.path.join(PROCESSED_DIR, filename)
            df = pd.read_csv(data_path)
            
            print(f"  Data shape: {df.shape}")
            print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
            
            # Create and fit stock scaler
            stock_scaler = MinMaxScaler()
            stock_scaler.fit(df[['Close (Rs.)']])
            
            # Create and fit macro scaler
            macro_scaler = MinMaxScaler()
            macro_scaler.fit(df[MACRO_FEATURES])
            
            # Save scalers
            stock_scaler_path = os.path.join(MODEL_DIR, f"{company_name}_stock_scaler.save")
            macro_scaler_path = os.path.join(MODEL_DIR, f"{company_name}_macro_scaler.save")
            
            joblib.dump(stock_scaler, stock_scaler_path)
            joblib.dump(macro_scaler, macro_scaler_path)
            
            print(f"  Stock scaler saved: {stock_scaler_path}")
            print(f"  Macro scaler saved: {macro_scaler_path}")
            
            # Test the scalers
            test_stock = stock_scaler.transform(df[['Close (Rs.)']].iloc[:5])
            test_macro = macro_scaler.transform(df[MACRO_FEATURES].iloc[:5])
            
            print(f"  Stock scaler test - shape: {test_stock.shape}")
            print(f"  Macro scaler test - shape: {test_macro.shape}")
            
        except Exception as e:
            print(f"  Error processing {filename}: {str(e)}")
            continue
    
    print(f"\nScaler regeneration completed!")

def test_scaler_loading():
    """Test if the regenerated scalers can be loaded"""
    print("\nTesting scaler loading...")
    
    if os.path.exists(MODEL_DIR):
        scaler_files = [f for f in os.listdir(MODEL_DIR) if f.endswith("_stock_scaler.save")]
        
        for scaler_file in scaler_files:
            company_name = scaler_file.replace("_stock_scaler.save", "")
            
            try:
                # Test stock scaler
                stock_scaler_path = os.path.join(MODEL_DIR, f"{company_name}_stock_scaler.save")
                stock_scaler = joblib.load(stock_scaler_path)
                
                # Test macro scaler
                macro_scaler_path = os.path.join(MODEL_DIR, f"{company_name}_macro_scaler.save")
                macro_scaler = joblib.load(macro_scaler_path)
                
                print(f"✓ Successfully loaded scalers for {company_name}")
                
            except Exception as e:
                print(f"✗ Failed to load scalers for {company_name}: {str(e)}")

if __name__ == "__main__":
    print("Starting scaler regeneration process...")
    regenerate_scalers()
    test_scaler_loading()
    print("\nScaler regeneration process completed!") 