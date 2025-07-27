#!/usr/bin/env python3
"""
Generate performance data for macro-economic models
This script creates the missing performance CSV files that the API needs
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model

# Constants
SEQUENCE_DIR = "data/macro-economic/data/processed/sequences"
MODEL_DIR = "models/macro-economic/models"
OUTPUT_DIR = "data/macro-economic/outputs"
MACRO_FEATURES = ['Tourism Arrivals', 'Money Supply', 'Exchange Rate', 'Inflation rate']

def generate_performance_data():
    """Generate performance data for all companies"""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Generating performance data for all companies...")
    
    # Get all available companies from sequence files
    companies = set()
    if os.path.exists(SEQUENCE_DIR):
        for fname in os.listdir(SEQUENCE_DIR):
            if fname.endswith("_X_stock.npy"):
                company = fname.replace("_X_stock.npy", "")
                companies.add(company)
    
    print(f"Found {len(companies)} companies")
    
    # Generate performance data for each company
    for company in sorted(companies):
        try:
            print(f"\nProcessing performance data for: {company}")
            
            # Load model and scalers
            model_path = os.path.join(MODEL_DIR, f"{company}_lstm_model.keras")
            h5_model_path = os.path.join(MODEL_DIR, f"{company}_lstm_model.h5")
            stock_scaler_path = os.path.join(MODEL_DIR, f"{company}_stock_scaler.save")
            
            # Try loading model
            model = None
            if os.path.exists(h5_model_path):
                try:
                    model = load_model(h5_model_path, compile=False)
                except:
                    pass
            
            if model is None and os.path.exists(model_path):
                try:
                    model = load_model(model_path, compile=False)
                except:
                    pass
            
            if model is None:
                print(f"  Skipping {company}: Could not load model")
                continue
            
            # Load scaler
            stock_scaler = joblib.load(stock_scaler_path)
            
            # Load sequences
            X_stock = np.load(os.path.join(SEQUENCE_DIR, f"{company}_X_stock.npy"))
            X_macro = np.load(os.path.join(SEQUENCE_DIR, f"{company}_X_macro.npy"))
            y_scaled = np.load(os.path.join(SEQUENCE_DIR, f"{company}_y.npy"))
            
            print(f"  Data shapes: X_stock={X_stock.shape}, X_macro={X_macro.shape}, y={y_scaled.shape}")
            
            # Make predictions on all data
            y_pred_scaled = model.predict([X_stock, X_macro])
            
            # Convert to actual prices
            y_pred = stock_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_actual = stock_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
            
            # Create performance dataframe
            performance_df = pd.DataFrame({
                "Actual Price": y_actual,
                "Predicted Price": y_pred
            })
            
            # Save performance data
            performance_file = os.path.join(OUTPUT_DIR, f"{company}_forecast_vs_actual.csv")
            performance_df.to_csv(performance_file, index=False)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
            mae = mean_absolute_error(y_actual, y_pred)
            mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
            
            print(f"  Performance metrics:")
            print(f"    RMSE: {rmse:.4f}")
            print(f"    MAE: {mae:.4f}")
            print(f"    MAPE: {mape:.2f}%")
            print(f"  Saved: {performance_file}")
            
        except Exception as e:
            print(f"  Error processing {company}: {str(e)}")
            continue
    
    print(f"\nPerformance data generation completed!")

def create_summary_metrics():
    """Create a summary CSV with all company metrics"""
    print("\nCreating summary metrics...")
    
    summary_data = []
    
    if os.path.exists(OUTPUT_DIR):
        for filename in os.listdir(OUTPUT_DIR):
            if filename.endswith("_forecast_vs_actual.csv"):
                company = filename.replace("_forecast_vs_actual.csv", "")
                
                try:
                    df = pd.read_csv(os.path.join(OUTPUT_DIR, filename))
                    
                    actual = df['Actual Price'].values
                    predicted = df['Predicted Price'].values
                    
                    rmse = np.sqrt(mean_squared_error(actual, predicted))
                    mae = mean_absolute_error(actual, predicted)
                    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
                    
                    summary_data.append({
                        "Company": company,
                        "RMSE": round(rmse, 4),
                        "MAE": round(mae, 4),
                        "MAPE": round(mape, 2),
                        "Data_Points": len(df)
                    })
                    
                except Exception as e:
                    print(f"  Error processing summary for {company}: {str(e)}")
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(OUTPUT_DIR, "summary_model_performance.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"  Summary saved: {summary_file}")
        
        # Print summary
        print("\nModel Performance Summary:")
        print(summary_df.to_string(index=False))

if __name__ == "__main__":
    print("Starting performance data generation...")
    generate_performance_data()
    create_summary_metrics()
    print("\nPerformance data generation process completed!") 