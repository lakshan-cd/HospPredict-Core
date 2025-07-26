"""
Utility functions for Macro-Economic Module (PyTorch Version)
Handles data processing, model training, and prediction functionality
Compatible with Python 3.13+
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Constants
MACRO_FEATURES = ['Tourism Arrivals', 'Money Supply', 'Exchange Rate', 'Inflation rate']
WINDOW_SIZE = 30
SEQUENCE_DIR = "data/macro-economic/data/processed/sequences"
MODEL_DIR = "models/macro-economic/models"
PROCESSED_DIR = "data/macro-economic/data/processed"
OUTPUT_DIR = "outputs"

class DualLSTM(nn.Module):
    """Dual-input LSTM model using PyTorch"""
    
    def __init__(self, stock_input_size=1, macro_input_size=4, hidden_size=64, output_size=1):
        super(DualLSTM, self).__init__()
        
        # Stock LSTM branch
        self.stock_lstm = nn.LSTM(stock_input_size, hidden_size, batch_first=True)
        
        # Macro LSTM branch
        self.macro_lstm = nn.LSTM(macro_input_size, hidden_size, batch_first=True)
        
        # Combined layers
        self.fc1 = nn.Linear(hidden_size * 2, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_size)
        
    def forward(self, stock_input, macro_input):
        # Stock branch
        stock_lstm_out, _ = self.stock_lstm(stock_input)
        stock_features = stock_lstm_out[:, -1, :]  # Take last time step
        
        # Macro branch
        macro_lstm_out, _ = self.macro_lstm(macro_input)
        macro_features = macro_lstm_out[:, -1, :]  # Take last time step
        
        # Combine features
        combined = torch.cat([stock_features, macro_features], dim=1)
        
        # Dense layers
        x = self.relu(self.fc1(combined))
        output = self.fc2(x)
        
        return output

class MacroEconomicProcessor:
    """Handles data processing for macro-economic analysis"""
    
    def __init__(self):
        self.stock_scaler = MinMaxScaler()
        self.macro_scaler = MinMaxScaler()
    
    def process_company_data(self, stock_df: pd.DataFrame, macro_df: pd.DataFrame, company_name: str) -> Dict[str, Any]:
        """
        Process stock and macro data for a company
        
        Args:
            stock_df: DataFrame with stock data
            macro_df: DataFrame with macro data
            company_name: Name of the company
            
        Returns:
            Dictionary containing processed data and file paths
        """
        try:
            # Clean company name
            clean_name = company_name.replace(" ", "_").replace("(", "").replace(")", "")
            
            # Process dates
            stock_df['Date'] = pd.to_datetime(stock_df['Trade Date'], format='%m/%d/%y', errors='coerce')
            macro_df['Date'] = pd.to_datetime(macro_df['Date'], errors='coerce')
            
            # Merge data
            merged_df = pd.merge(stock_df, macro_df, on='Date', how='inner')
            merged_df.sort_values('Date', inplace=True)
            
            if len(merged_df) < 60:
                raise ValueError(f"Insufficient data points for {company_name}. Minimum 60 required, got {len(merged_df)}")
            
            # Apply scaling
            merged_df['Close_scaled'] = self.stock_scaler.fit_transform(merged_df[['Close (Rs.)']])
            merged_df[MACRO_FEATURES] = self.macro_scaler.fit_transform(merged_df[MACRO_FEATURES])
            
            # Save processed data
            os.makedirs(PROCESSED_DIR, exist_ok=True)
            processed_file = os.path.join(PROCESSED_DIR, f"{clean_name}_scaled.csv")
            merged_df.to_csv(processed_file, index=False)
            
            # Save scalers
            os.makedirs(MODEL_DIR, exist_ok=True)
            joblib.dump(self.stock_scaler, os.path.join(MODEL_DIR, f"{clean_name}_stock_scaler.save"))
            joblib.dump(self.macro_scaler, os.path.join(MODEL_DIR, f"{clean_name}_macro_scaler.save"))
            
            return {
                "company_name": clean_name,
                "processed_file": processed_file,
                "data_points": len(merged_df),
                "date_range": {
                    "start": merged_df['Date'].min().strftime("%Y-%m-%d"),
                    "end": merged_df['Date'].max().strftime("%Y-%m-%d")
                }
            }
            
        except Exception as e:
            raise Exception(f"Error processing data for {company_name}: {str(e)}")
    
    def create_sequences(self, company_name: str) -> Dict[str, Any]:
        """
        Create LSTM sequences from processed data
        
        Args:
            company_name: Name of the company
            
        Returns:
            Dictionary containing sequence information
        """
        try:
            # Load processed data
            processed_file = os.path.join(PROCESSED_DIR, f"{company_name}_scaled.csv")
            if not os.path.exists(processed_file):
                raise FileNotFoundError(f"Processed data not found for {company_name}")
            
            df = pd.read_csv(processed_file)
            
            if len(df) <= WINDOW_SIZE:
                raise ValueError(f"Insufficient data for sequences. Need more than {WINDOW_SIZE} points")
            
            # Create sequences
            X_stock = []
            X_macro = []
            y = []
            
            for i in range(WINDOW_SIZE, len(df)):
                X_stock.append(df['Close_scaled'].values[i-WINDOW_SIZE:i])
                X_macro.append(df[MACRO_FEATURES].values[i-WINDOW_SIZE:i])
                y.append(df['Close_scaled'].values[i])
            
            X_stock = np.array(X_stock)[..., np.newaxis]  # Shape: (samples, 30, 1)
            X_macro = np.array(X_macro)                  # Shape: (samples, 30, 4)
            y = np.array(y)
            
            # Save sequences
            os.makedirs(SEQUENCE_DIR, exist_ok=True)
            np.save(os.path.join(SEQUENCE_DIR, f"{company_name}_X_stock.npy"), X_stock)
            np.save(os.path.join(SEQUENCE_DIR, f"{company_name}_X_macro.npy"), X_macro)
            np.save(os.path.join(SEQUENCE_DIR, f"{company_name}_y.npy"), y)
            
            return {
                "company_name": company_name,
                "sequences_created": True,
                "X_stock_shape": X_stock.shape,
                "X_macro_shape": X_macro.shape,
                "y_shape": y.shape,
                "total_samples": len(y)
            }
            
        except Exception as e:
            raise Exception(f"Error creating sequences for {company_name}: {str(e)}")

class MacroEconomicTrainer:
    """Handles model training for macro-economic analysis using PyTorch"""
    
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def build_model(self, stock_input_size: int = 1, macro_input_size: int = 4) -> DualLSTM:
        """
        Build dual-input LSTM model
        
        Args:
            stock_input_size: Size of stock input features
            macro_input_size: Size of macro input features
            
        Returns:
            PyTorch model
        """
        model = DualLSTM(
            stock_input_size=stock_input_size,
            macro_input_size=macro_input_size,
            hidden_size=64,
            output_size=1
        ).to(self.device)
        
        return model
    
    def train_model(self, company_name: str, epochs: int = 50, batch_size: int = 32) -> Dict[str, Any]:
        """
        Train LSTM model for a company
        
        Args:
            company_name: Name of the company
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary containing training results
        """
        try:
            # Load sequences
            X_stock = np.load(os.path.join(SEQUENCE_DIR, f"{company_name}_X_stock.npy"))
            X_macro = np.load(os.path.join(SEQUENCE_DIR, f"{company_name}_X_macro.npy"))
            y = np.load(os.path.join(SEQUENCE_DIR, f"{company_name}_y.npy"))
            
            # Convert to PyTorch tensors
            X_stock_tensor = torch.FloatTensor(X_stock).to(self.device)
            X_macro_tensor = torch.FloatTensor(X_macro).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)
            
            # Train/test split
            split_idx = int(0.8 * len(y))
            X_stock_train = X_stock_tensor[:split_idx]
            X_stock_test = X_stock_tensor[split_idx:]
            X_macro_train = X_macro_tensor[:split_idx]
            X_macro_test = X_macro_tensor[split_idx:]
            y_train = y_tensor[:split_idx]
            y_test = y_tensor[split_idx:]
            
            # Build model
            self.model = self.build_model()
            
            # Loss function and optimizer
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            
            # Training loop
            train_losses = []
            val_losses = []
            
            for epoch in range(epochs):
                # Training
                self.model.train()
                optimizer.zero_grad()
                
                outputs = self.model(X_stock_train, X_macro_train)
                loss = criterion(outputs.squeeze(), y_train)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
                
                # Validation
                if epoch % 10 == 0:
                    self.model.eval()
                    with torch.no_grad():
                        val_outputs = self.model(X_stock_test, X_macro_test)
                        val_loss = criterion(val_outputs.squeeze(), y_test)
                        val_losses.append(val_loss.item())
                        
                        if len(val_losses) > 1 and val_losses[-1] > val_losses[-2]:
                            # Early stopping
                            break
            
            # Save model
            model_path = os.path.join(MODEL_DIR, f"{company_name}_lstm_model.pth")
            torch.save(self.model.state_dict(), model_path)
            
            # Evaluate model
            self.model.eval()
            with torch.no_grad():
                y_pred = self.model(X_stock_test, X_macro_test).cpu().numpy().flatten()
                y_test_np = y_test.cpu().numpy()
                
                rmse = np.sqrt(mean_squared_error(y_test_np, y_pred))
                mae = mean_absolute_error(y_test_np, y_pred)
            
            return {
                "company_name": company_name,
                "model_saved": True,
                "model_path": model_path,
                "training_epochs": len(train_losses),
                "final_loss": train_losses[-1],
                "test_rmse": rmse,
                "test_mae": mae
            }
            
        except Exception as e:
            raise Exception(f"Error training model for {company_name}: {str(e)}")

class MacroEconomicPredictor:
    """Handles predictions using trained PyTorch models"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_model(self, company_name: str) -> bool:
        """
        Load trained model and scalers for a company
        
        Args:
            company_name: Name of the company
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Load model
            model_path = os.path.join(MODEL_DIR, f"{company_name}_lstm_model.pth")
            if not os.path.exists(model_path):
                return False
            
            model = DualLSTM().to(self.device)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            self.models[company_name] = model
            
            # Load scalers
            stock_scaler_path = os.path.join(MODEL_DIR, f"{company_name}_stock_scaler.save")
            macro_scaler_path = os.path.join(MODEL_DIR, f"{company_name}_macro_scaler.save")
            
            if os.path.exists(stock_scaler_path) and os.path.exists(macro_scaler_path):
                self.scalers[company_name] = {
                    'stock': joblib.load(stock_scaler_path),
                    'macro': joblib.load(macro_scaler_path)
                }
            
            return True
            
        except Exception as e:
            print(f"Error loading model for {company_name}: {str(e)}")
            return False
    
    def predict(self, company_name: str, days_ahead: int = 1) -> Dict[str, Any]:
        """
        Make prediction for a company
        
        Args:
            company_name: Name of the company
            days_ahead: Number of days to predict ahead
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            if company_name not in self.models:
                if not self.load_model(company_name):
                    raise Exception(f"Model not found for {company_name}")
            
            # Load latest sequences
            X_stock = np.load(os.path.join(SEQUENCE_DIR, f"{company_name}_X_stock.npy"))
            X_macro = np.load(os.path.join(SEQUENCE_DIR, f"{company_name}_X_macro.npy"))
            
            # Get the most recent sequence
            latest_stock = torch.FloatTensor(X_stock[-1:]).to(self.device)
            latest_macro = torch.FloatTensor(X_macro[-1:]).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                prediction_scaled = self.models[company_name](latest_stock, latest_macro)
                prediction = prediction_scaled.cpu().numpy().flatten()[0]
            
            # Inverse transform
            prediction_actual = self.scalers[company_name]['stock'].inverse_transform(
                [[prediction]]
            )[0, 0]
            
            # Get current price for comparison
            current_price_scaled = X_stock[-1, -1, 0]
            current_price = self.scalers[company_name]['stock'].inverse_transform(
                [[current_price_scaled]]
            )[0, 0]
            
            return {
                "company_name": company_name,
                "current_price": round(current_price, 2),
                "predicted_price": round(prediction_actual, 2),
                "price_change": round(prediction_actual - current_price, 2),
                "price_change_percent": round(((prediction_actual - current_price) / current_price) * 100, 2),
                "prediction_date": datetime.now().strftime("%Y-%m-%d"),
                "days_ahead": days_ahead
            }
            
        except Exception as e:
            raise Exception(f"Error making prediction for {company_name}: {str(e)}")

def get_available_companies() -> List[str]:
    """Get list of available companies with trained models"""
    companies = []
    if os.path.exists(MODEL_DIR):
        for file in os.listdir(MODEL_DIR):
            if file.endswith("_lstm_model.pth"):
                company = file.replace("_lstm_model.pth", "")
                companies.append(company)
    return sorted(companies)

def validate_company_data(stock_data: List[Dict], macro_data: List[Dict]) -> Tuple[bool, str]:
    """
    Validate company data format
    
    Args:
        stock_data: List of stock data dictionaries
        macro_data: List of macro data dictionaries
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check stock data
        if not stock_data:
            return False, "Stock data is empty"
        
        required_stock_fields = ['Trade Date', 'Close (Rs.)']
        for field in required_stock_fields:
            if field not in stock_data[0]:
                return False, f"Stock data missing required field: {field}"
        
        # Check macro data
        if not macro_data:
            return False, "Macro data is empty"
        
        for field in MACRO_FEATURES:
            if field not in macro_data[0]:
                return False, f"Macro data missing required field: {field}"
        
        # Check minimum data points
        if len(stock_data) < 60:
            return False, f"Insufficient stock data points. Minimum 60 required, got {len(stock_data)}"
        
        if len(macro_data) < 60:
            return False, f"Insufficient macro data points. Minimum 60 required, got {len(macro_data)}"
        
        return True, "Data validation passed"
        
    except Exception as e:
        return False, f"Data validation error: {str(e)}" 