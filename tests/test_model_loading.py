#!/usr/bin/env python3
"""
Test script to diagnose Keras model loading issues
"""

import os
import sys
import traceback
import numpy as np
import joblib
from keras.models import load_model
import tensorflow as tf

# Constants
MODEL_DIR = "models/macro-economic/models"
SEQUENCE_DIR = "data/macro-economic/data/processed/sequences"
MACRO_FEATURES = ['Tourism Arrivals', 'Money Supply', 'Exchange Rate', 'Inflation rate']
WINDOW_SIZE = 30

def test_model_loading(company_name: str):
    """Test different approaches to load the model"""
    print(f"Testing model loading for: {company_name}")
    
    # Check if files exist
    model_path = os.path.join(MODEL_DIR, f"{company_name}_lstm_model.keras")
    h5_model_path = os.path.join(MODEL_DIR, f"{company_name}_lstm_model.h5")
    stock_scaler_path = os.path.join(MODEL_DIR, f"{company_name}_stock_scaler.save")
    macro_scaler_path = os.path.join(MODEL_DIR, f"{company_name}_macro_scaler.save")
    
    print(f"Model files exist:")
    print(f"  .keras: {os.path.exists(model_path)}")
    print(f"  .h5: {os.path.exists(h5_model_path)}")
    print(f"  stock_scaler: {os.path.exists(stock_scaler_path)}")
    print(f"  macro_scaler: {os.path.exists(macro_scaler_path)}")
    
    # Test 1: Basic load_model
    print("\nTest 1: Basic load_model")
    try:
        model = load_model(model_path, compile=False)
        print("✓ Success: Basic load_model")
        return model
    except Exception as e:
        print(f"✗ Failed: {str(e)}")
    
    # Test 2: Load with custom_objects
    print("\nTest 2: Load with custom_objects")
    try:
        model = load_model(model_path, compile=False, custom_objects={})
        print("✓ Success: Load with custom_objects")
        return model
    except Exception as e:
        print(f"✗ Failed: {str(e)}")
    
    # Test 3: Load with safe_mode=False
    print("\nTest 3: Load with safe_mode=False")
    try:
        model = load_model(model_path, compile=False, safe_mode=False)
        print("✓ Success: Load with safe_mode=False")
        return model
    except Exception as e:
        print(f"✗ Failed: {str(e)}")
    
    # Test 4: Load .h5 file
    print("\nTest 4: Load .h5 file")
    try:
        if os.path.exists(h5_model_path):
            model = load_model(h5_model_path, compile=False)
            print("✓ Success: Load .h5 file")
            return model
        else:
            print("✗ Failed: .h5 file not found")
    except Exception as e:
        print(f"✗ Failed: {str(e)}")
    
    # Test 5: Clear session and try again
    print("\nTest 5: Clear session and try again")
    try:
        tf.keras.backend.clear_session()
        model = load_model(model_path, compile=False)
        print("✓ Success: Clear session and load")
        return model
    except Exception as e:
        print(f"✗ Failed: {str(e)}")
    
    # Test 6: Try with different TensorFlow settings
    print("\nTest 6: Try with different TensorFlow settings")
    try:
        tf.config.experimental.enable_tensor_float_32_execution(False)
        model = load_model(model_path, compile=False)
        print("✓ Success: Different TensorFlow settings")
        return model
    except Exception as e:
        print(f"✗ Failed: {str(e)}")
    
    print("\nAll loading attempts failed!")
    return None

def test_prediction(company_name: str):
    """Test making a prediction with the loaded model"""
    print(f"\nTesting prediction for: {company_name}")
    
    # Load model
    model = test_model_loading(company_name)
    if model is None:
        print("Cannot test prediction - model loading failed")
        return
    
    # Load scalers
    try:
        stock_scaler_path = os.path.join(MODEL_DIR, f"{company_name}_stock_scaler.save")
        macro_scaler_path = os.path.join(MODEL_DIR, f"{company_name}_macro_scaler.save")
        
        stock_scaler = joblib.load(stock_scaler_path)
        macro_scaler = joblib.load(macro_scaler_path)
        print("✓ Scalers loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load scalers: {str(e)}")
        return
    
    # Load sequences
    try:
        X_stock = np.load(os.path.join(SEQUENCE_DIR, f"{company_name}_X_stock.npy"))
        X_macro = np.load(os.path.join(SEQUENCE_DIR, f"{company_name}_X_macro.npy"))
        print("✓ Sequences loaded successfully")
        print(f"  X_stock shape: {X_stock.shape}")
        print(f"  X_macro shape: {X_macro.shape}")
    except Exception as e:
        print(f"✗ Failed to load sequences: {str(e)}")
        return
    
    # Make prediction
    try:
        # Get the most recent sequence
        latest_stock = X_stock[-1:].reshape(1, WINDOW_SIZE, 1)
        latest_macro = X_macro[-1:].reshape(1, WINDOW_SIZE, len(MACRO_FEATURES))
        
        print(f"  Input shapes: stock={latest_stock.shape}, macro={latest_macro.shape}")
        
        # Make prediction
        prediction_scaled = model.predict([latest_stock, latest_macro])
        prediction = stock_scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()[0]
        
        # Get current price for comparison
        current_price_scaled = X_stock[-1, -1, 0]
        current_price = stock_scaler.inverse_transform([[current_price_scaled]])[0, 0]
        
        print("✓ Prediction successful!")
        print(f"  Current price: {current_price:.2f}")
        print(f"  Predicted price: {prediction:.2f}")
        print(f"  Price change: {prediction - current_price:.2f}")
        
    except Exception as e:
        print(f"✗ Failed to make prediction: {str(e)}")
        print(f"  Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    # Test with the company that's failing
    company_name = "AITKEN_SPENCE_HOTEL_HOLDINGS_PLC"
    test_prediction(company_name) 