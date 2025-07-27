#!/usr/bin/env python3
"""
Retrain LSTM models for macro-economic prediction
This script retrains all models in the current environment to ensure compatibility
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from sklearn.preprocessing import MinMaxScaler

# Constants
SEQUENCE_DIR = "data/macro-economic/data/processed/sequences"
MODEL_DIR = "models/macro-economic/models"
PROCESSED_DIR = "data/macro-economic/data/processed"
MACRO_FEATURES = ['Tourism Arrivals', 'Money Supply', 'Exchange Rate', 'Inflation rate']
WINDOW_SIZE = 30

def retrain_models():
    """Retrain all LSTM models for compatibility"""
    
    # Create model directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Get all available companies from sequence files
    companies = set()
    if os.path.exists(SEQUENCE_DIR):
        for fname in os.listdir(SEQUENCE_DIR):
            if fname.endswith("_X_stock.npy"):
                company = fname.replace("_X_stock.npy", "")
                companies.add(company)
    
    print(f"Found {len(companies)} companies to retrain")
    
    # Train models for each company
    for company in sorted(companies):
        try:
            print(f"\nTraining model for: {company}")
            
            # Load sequences
            X_stock = np.load(os.path.join(SEQUENCE_DIR, f"{company}_X_stock.npy"))
            X_macro = np.load(os.path.join(SEQUENCE_DIR, f"{company}_X_macro.npy"))
            y = np.load(os.path.join(SEQUENCE_DIR, f"{company}_y.npy"))
            
            print(f"  Data shapes: X_stock={X_stock.shape}, X_macro={X_macro.shape}, y={y.shape}")
            
            # Train/test split (80/20)
            split_idx = int(0.8 * len(y))
            X_stock_train, X_stock_test = X_stock[:split_idx], X_stock[split_idx:]
            X_macro_train, X_macro_test = X_macro[:split_idx], X_macro[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Build dual-input LSTM model
            input_stock = Input(shape=(X_stock.shape[1], 1), name="Stock_Input")
            input_macro = Input(shape=(X_macro.shape[1], X_macro.shape[2]), name="Macro_Input")
            
            x1 = LSTM(64, return_sequences=False)(input_stock)
            x2 = LSTM(64, return_sequences=False)(input_macro)
            
            x = Concatenate()([x1, x2])
            x = Dense(32, activation='relu')(x)
            output = Dense(1, name='Forecast')(x)
            
            model = Model(inputs=[input_stock, input_macro], outputs=output)
            model.compile(optimizer=Adam(0.001), loss="mse")
            
            # Early stopping callback
            early_stop = EarlyStopping(
                monitor='val_loss', 
                patience=5, 
                restore_best_weights=True,
                verbose=0
            )
            
            # Train the model
            history = model.fit(
                [X_stock_train, X_macro_train],
                y_train,
                validation_split=0.1,
                epochs=50,
                batch_size=32,
                callbacks=[early_stop],
                verbose=0
            )
            
            # Save model in both formats for compatibility
            model_path_keras = os.path.join(MODEL_DIR, f"{company}_lstm_model.keras")
            model_path_h5 = os.path.join(MODEL_DIR, f"{company}_lstm_model.h5")
            
            model.save(model_path_keras)
            model.save(model_path_h5)
            
            print(f"  Model saved: {model_path_keras}")
            print(f"  Model saved: {model_path_h5}")
            
            # Evaluate model
            y_pred = model.predict([X_stock_test, X_macro_test])
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            print(f"  Test RMSE: {rmse:.4f}")
            print(f"  Test MAE: {mae:.4f}")
            
            # Check if scalers exist, if not create them
            stock_scaler_path = os.path.join(MODEL_DIR, f"{company}_stock_scaler.save")
            macro_scaler_path = os.path.join(MODEL_DIR, f"{company}_macro_scaler.save")
            
            if not os.path.exists(stock_scaler_path) or not os.path.exists(macro_scaler_path):
                print(f"  Creating scalers for {company}")
                
                # Load processed data to recreate scalers
                processed_file = os.path.join(PROCESSED_DIR, f"{company}_scaled.csv")
                if os.path.exists(processed_file):
                    df = pd.read_csv(processed_file)
                    
                    # Create and save stock scaler
                    stock_scaler = MinMaxScaler()
                    stock_scaler.fit(df[['Close (Rs.)']])
                    joblib.dump(stock_scaler, stock_scaler_path)
                    
                    # Create and save macro scaler
                    macro_scaler = MinMaxScaler()
                    macro_scaler.fit(df[MACRO_FEATURES])
                    joblib.dump(macro_scaler, macro_scaler_path)
                    
                    print(f"  Scalers created and saved")
            
        except Exception as e:
            print(f"  Error training {company}: {str(e)}")
            continue
    
    print(f"\nModel retraining completed!")

def test_model_loading():
    """Test if the retrained models can be loaded"""
    print("\nTesting model loading...")
    
    if os.path.exists(MODEL_DIR):
        for fname in os.listdir(MODEL_DIR):
            if fname.endswith("_lstm_model.keras"):
                company = fname.replace("_lstm_model.keras", "")
                model_path = os.path.join(MODEL_DIR, fname)
                
                try:
                    from tensorflow.keras.models import load_model
                    model = load_model(model_path, compile=False)
                    print(f"✓ Successfully loaded model for {company}")
                except Exception as e:
                    print(f"✗ Failed to load model for {company}: {str(e)}")

if __name__ == "__main__":
    print("Starting model retraining process...")
    retrain_models()
    test_model_loading()
    print("\nRetraining process completed!") 