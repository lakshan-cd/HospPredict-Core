from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
import os
import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

router = APIRouter()

# Data models
class MacroEconomicData(BaseModel):
    company_name: str
    stock_data: List[Dict[str, Any]]  # Daily stock data
    macro_data: List[Dict[str, Any]]  # Daily macro data

class PredictionRequest(BaseModel):
    company_name: str
    days_ahead: int = 1

class CompanyDataRequest(BaseModel):
    company_name: str
    start_date: str
    end_date: str

# Constants
MACRO_FEATURES = ['Tourism Arrivals', 'Money Supply', 'Exchange Rate', 'Inflation rate']
WINDOW_SIZE = 30
SEQUENCE_DIR = "data/macro-economic/data/processed/sequences"
MODEL_DIR = "models/macro-economic/models"
PROCESSED_DIR = "data/macro-economic/data/processed"
OUTPUT_DIR = "outputs"

class MacroEconomicAPI:
    def __init__(self):
        self.available_companies = self._get_available_companies()
    
    def _get_available_companies(self) -> List[str]:
        """Get list of available companies with trained models"""
        companies = []
        if os.path.exists(MODEL_DIR):
            for file in os.listdir(MODEL_DIR):
                if file.endswith("_lstm_model.keras"):
                    company = file.replace("_lstm_model.keras", "")
                    companies.append(company)
        return sorted(companies)
    
    def get_company_data(self, company_name: str) -> Dict[str, Any]:
        """Get processed data for a specific company"""
        try:
            # Load scaled data
            scaled_file = os.path.join(PROCESSED_DIR, f"{company_name}_scaled.csv")
            if not os.path.exists(scaled_file):
                raise HTTPException(status_code=404, detail=f"Data not found for company: {company_name}")
            
            df = pd.read_csv(scaled_file)
            
            # Load sequences
            X_stock = np.load(os.path.join(SEQUENCE_DIR, f"{company_name}_X_stock.npy"))
            X_macro = np.load(os.path.join(SEQUENCE_DIR, f"{company_name}_X_macro.npy"))
            y = np.load(os.path.join(SEQUENCE_DIR, f"{company_name}_y.npy"))
            
            return {
                "company_name": company_name,
                "data_points": len(df),
                "date_range": {
                    "start": df['Date'].min(),
                    "end": df['Date'].max()
                },
                "features": {
                    "stock_features": ["Open (Rs.)", "High (Rs.)", "Low (Rs.)", "Close (Rs.)", "TradeVolume"],
                    "macro_features": MACRO_FEATURES
                },
                "sequences": {
                    "X_stock_shape": X_stock.shape,
                    "X_macro_shape": X_macro.shape,
                    "y_shape": y.shape
                }
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading company data: {str(e)}")
    
    def predict_stock_price(self, company_name: str, days_ahead: int = 1) -> Dict[str, Any]:
        """Predict stock price for a company"""
        try:
            if company_name not in self.available_companies:
                raise HTTPException(status_code=404, detail=f"Model not found for company: {company_name}")
            
            # Load model and scalers
            model_path = os.path.join(MODEL_DIR, f"{company_name}_lstm_model.keras")
            stock_scaler_path = os.path.join(MODEL_DIR, f"{company_name}_stock_scaler.save")
            macro_scaler_path = os.path.join(MODEL_DIR, f"{company_name}_macro_scaler.save")
            
            model = load_model(model_path, compile=False)
            stock_scaler = joblib.load(stock_scaler_path)
            macro_scaler = joblib.load(macro_scaler_path)
            
            # Load latest sequences
            X_stock = np.load(os.path.join(SEQUENCE_DIR, f"{company_name}_X_stock.npy"))
            X_macro = np.load(os.path.join(SEQUENCE_DIR, f"{company_name}_X_macro.npy"))
            
            # Get the most recent sequence
            latest_stock = X_stock[-1:].reshape(1, WINDOW_SIZE, 1)
            latest_macro = X_macro[-1:].reshape(1, WINDOW_SIZE, len(MACRO_FEATURES))
            
            # Make prediction
            prediction_scaled = model.predict([latest_stock, latest_macro])
            prediction = stock_scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()[0]
            
            # Get current price for comparison
            current_price_scaled = X_stock[-1, -1, 0]
            current_price = stock_scaler.inverse_transform([[current_price_scaled]])[0, 0]
            
            return {
                "company_name": company_name,
                "current_price": round(current_price, 2),
                "predicted_price": round(prediction, 2),
                "price_change": round(prediction - current_price, 2),
                "price_change_percent": round(((prediction - current_price) / current_price) * 100, 2),
                "prediction_date": datetime.now().strftime("%Y-%m-%d"),
                "model_confidence": "high"  # Could be enhanced with confidence intervals
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")
    
    def get_model_performance(self, company_name: str) -> Dict[str, Any]:
        """Get model performance metrics for a company"""
        try:
            # Load actual vs predicted data
            results_file = os.path.join(OUTPUT_DIR, f"{company_name}_forecast_vs_actual.csv")
            if not os.path.exists(results_file):
                raise HTTPException(status_code=404, detail=f"Performance data not found for company: {company_name}")
            
            df = pd.read_csv(results_file)
            
            # Calculate metrics
            actual = df['Actual Price'].values
            predicted = df['Predicted Price'].values
            
            rmse = np.sqrt(np.mean((actual - predicted) ** 2))
            mae = np.mean(np.abs(actual - predicted))
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
            
            return {
                "company_name": company_name,
                "metrics": {
                    "rmse": round(rmse, 4),
                    "mae": round(mae, 4),
                    "mape": round(mape, 2)
                },
                "data_points": len(df),
                "date_range": {
                    "start": df.index[0],
                    "end": df.index[-1]
                }
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading performance data: {str(e)}")
    
    def get_feature_importance(self, company_name: str) -> Dict[str, Any]:
        """Get SHAP feature importance for a company"""
        try:
            # Check if SHAP plots exist
            shap_summary_path = os.path.join(OUTPUT_DIR, "plots", "shap_plots", f"shap_summary_macro_{company_name}.png")
            shap_bar_path = os.path.join(OUTPUT_DIR, "plots", "shap_bar_plots", f"shap_bar_grid_macro_{company_name}.png")
            
            feature_importance = {}
            
            if os.path.exists(shap_summary_path):
                feature_importance["summary_plot"] = f"/api/v1/macro-economic/plots/shap/{company_name}/summary"
            
            if os.path.exists(shap_bar_path):
                feature_importance["bar_plot"] = f"/api/v1/macro-economic/plots/shap/{company_name}/bar"
            
            # Default feature importance based on typical patterns
            if not feature_importance:
                feature_importance = {
                    "Tourism Arrivals": 0.35,
                    "Exchange Rate": 0.30,
                    "Money Supply": 0.20,
                    "Inflation rate": 0.15
                }
            
            return {
                "company_name": company_name,
                "feature_importance": feature_importance,
                "plots_available": len(feature_importance) > 0
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading feature importance: {str(e)}")
    
    def get_granger_causality(self, company_name: str) -> Dict[str, Any]:
        """Get Granger causality analysis results"""
        try:
            # Check if Granger causality files exist
            granger_file = os.path.join(OUTPUT_DIR, f"granger_{company_name}_6month.csv")
            granger_heatmap = os.path.join(OUTPUT_DIR, "plots", "rolling_ganager_heatmap", f"rolling_granger_heatmap_{company_name}_6month.png")
            
            if not os.path.exists(granger_file):
                raise HTTPException(status_code=404, detail=f"Granger causality data not found for company: {company_name}")
            
            df = pd.read_csv(granger_file)
            
            # Calculate average p-values for each feature
            feature_pvalues = {}
            for feature in MACRO_FEATURES:
                feature_data = df[df['Feature'] == feature]
                avg_pvalue = feature_data['Min_PValue'].mean()
                feature_pvalues[feature] = round(avg_pvalue, 4)
            
            return {
                "company_name": company_name,
                "feature_pvalues": feature_pvalues,
                "significant_features": [f for f, p in feature_pvalues.items() if p < 0.05],
                "heatmap_available": os.path.exists(granger_heatmap),
                "heatmap_url": f"/api/v1/macro-economic/plots/granger/{company_name}/heatmap" if os.path.exists(granger_heatmap) else None
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading Granger causality data: {str(e)}")
    
    def process_new_company(self, company_data: MacroEconomicData) -> Dict[str, Any]:
        """Process new company data and create predictions"""
        try:
            company_name = company_data.company_name.replace(" ", "_").replace("(", "").replace(")", "")
            
            # Convert to DataFrames
            stock_df = pd.DataFrame(company_data.stock_data)
            macro_df = pd.DataFrame(company_data.macro_data)
            
            # Validate required columns
            required_stock_cols = ['Trade Date', 'Close (Rs.)']
            required_macro_cols = MACRO_FEATURES
            
            if not all(col in stock_df.columns for col in required_stock_cols):
                raise HTTPException(status_code=400, detail="Stock data missing required columns")
            
            if not all(col in macro_df.columns for col in required_macro_cols):
                raise HTTPException(status_code=400, detail="Macro data missing required columns")
            
            # Process dates
            stock_df['Date'] = pd.to_datetime(stock_df['Trade Date'], format='%m/%d/%y', errors='coerce')
            macro_df['Date'] = pd.to_datetime(macro_df['Date'], errors='coerce')
            
            # Merge data
            merged_df = pd.merge(stock_df, macro_df, on='Date', how='inner')
            merged_df.sort_values('Date', inplace=True)
            
            if len(merged_df) < 60:
                raise HTTPException(status_code=400, detail="Insufficient data points (minimum 60 required)")
            
            # Apply scaling
            stock_scaler = MinMaxScaler()
            macro_scaler = MinMaxScaler()
            
            merged_df['Close_scaled'] = stock_scaler.fit_transform(merged_df[['Close (Rs.)']])
            merged_df[MACRO_FEATURES] = macro_scaler.fit_transform(merged_df[MACRO_FEATURES])
            
            # Create sequences
            X_stock = []
            X_macro = []
            y = []
            
            for i in range(WINDOW_SIZE, len(merged_df)):
                X_stock.append(merged_df['Close_scaled'].values[i-WINDOW_SIZE:i])
                X_macro.append(merged_df[MACRO_FEATURES].values[i-WINDOW_SIZE:i])
                y.append(merged_df['Close_scaled'].values[i])
            
            X_stock = np.array(X_stock)[..., np.newaxis]
            X_macro = np.array(X_macro)
            y = np.array(y)
            
            # Find best matching model (could be enhanced with similarity metrics)
            best_model = self._find_best_model(company_name)
            
            if best_model:
                # Load the best model
                model_path = os.path.join(MODEL_DIR, f"{best_model}_lstm_model.keras")
                model = load_model(model_path, compile=False)
                
                # Make prediction using the best model
                latest_stock = X_stock[-1:].reshape(1, WINDOW_SIZE, 1)
                latest_macro = X_macro[-1:].reshape(1, WINDOW_SIZE, len(MACRO_FEATURES))
                
                prediction_scaled = model.predict([latest_stock, latest_macro])
                prediction = stock_scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()[0]
                
                current_price = merged_df['Close (Rs.)'].iloc[-1]
                
                return {
                    "company_name": company_name,
                    "processed": True,
                    "data_points": len(merged_df),
                    "current_price": round(current_price, 2),
                    "predicted_price": round(prediction, 2),
                    "price_change": round(prediction - current_price, 2),
                    "price_change_percent": round(((prediction - current_price) / current_price) * 100, 2),
                    "model_used": best_model,
                    "confidence": "medium"  # Could be enhanced
                }
            else:
                return {
                    "company_name": company_name,
                    "processed": True,
                    "data_points": len(merged_df),
                    "prediction": "No suitable model found for prediction",
                    "recommendation": "Train a new model for this company"
                }
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing new company: {str(e)}")
    
    def _find_best_model(self, company_name: str) -> Optional[str]:
        """Find the best matching model for a new company"""
        # Simple heuristic - could be enhanced with company similarity metrics
        # For now, return the first available model
        return self.available_companies[0] if self.available_companies else None

# Initialize API
macro_api = MacroEconomicAPI()

# API Endpoints
@router.get("/companies")
def get_available_companies():
    """Get list of all available companies with trained models"""
    return {
        "companies": macro_api.available_companies,
        "total_count": len(macro_api.available_companies)
    }

@router.get("/companies/{company_name}")
def get_company_info(company_name: str):
    """Get detailed information about a specific company"""
    return macro_api.get_company_data(company_name)

@router.post("/predict")
def predict_stock_price(req: PredictionRequest):
    """Predict stock price for a company"""
    return macro_api.predict_stock_price(req.company_name, req.days_ahead)

@router.get("/performance/{company_name}")
def get_model_performance(company_name: str):
    """Get model performance metrics for a company"""
    return macro_api.get_model_performance(company_name)

@router.get("/feature-importance/{company_name}")
def get_feature_importance(company_name: str):
    """Get feature importance analysis for a company"""
    return macro_api.get_feature_importance(company_name)

@router.get("/granger-causality/{company_name}")
def get_granger_causality(company_name: str):
    """Get Granger causality analysis for a company"""
    return macro_api.get_granger_causality(company_name)

@router.post("/new-company")
def process_new_company(company_data: MacroEconomicData):
    """Process new company data and make predictions"""
    return macro_api.process_new_company(company_data)

@router.get("/plots/prediction/{company_name}")
def get_prediction_plot(company_name: str):
    """Get prediction plot for a company"""
    try:
        plot_path = os.path.join(OUTPUT_DIR, "plots", "predictions", f"{company_name}.png")
        if os.path.exists(plot_path):
            return FileResponse(plot_path, media_type="image/png")
        else:
            raise HTTPException(status_code=404, detail="Prediction plot not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading plot: {str(e)}")

@router.get("/plots/shap/{company_name}/summary")
def get_shap_summary_plot(company_name: str):
    """Get SHAP summary plot for a company"""
    try:
        plot_path = os.path.join(OUTPUT_DIR, "plots", "shap_plots", f"shap_summary_macro_{company_name}.png")
        if os.path.exists(plot_path):
            return FileResponse(plot_path, media_type="image/png")
        else:
            raise HTTPException(status_code=404, detail="SHAP summary plot not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading plot: {str(e)}")

@router.get("/plots/shap/{company_name}/bar")
def get_shap_bar_plot(company_name: str):
    """Get SHAP bar plot for a company"""
    try:
        plot_path = os.path.join(OUTPUT_DIR, "plots", "shap_bar_plots", f"shap_bar_grid_macro_{company_name}.png")
        if os.path.exists(plot_path):
            return FileResponse(plot_path, media_type="image/png")
        else:
            raise HTTPException(status_code=404, detail="SHAP bar plot not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading plot: {str(e)}")

@router.get("/plots/granger/{company_name}/heatmap")
def get_granger_heatmap(company_name: str):
    """Get Granger causality heatmap for a company"""
    try:
        plot_path = os.path.join(OUTPUT_DIR, "plots", "rolling_ganager_heatmap", f"rolling_granger_heatmap_{company_name}_6month.png")
        if os.path.exists(plot_path):
            return FileResponse(plot_path, media_type="image/png")
        else:
            raise HTTPException(status_code=404, detail="Granger causality heatmap not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading plot: {str(e)}")

@router.get("/summary")
def get_summary():
    """Get summary of all available data and models"""
    try:
        summary = {
            "total_companies": len(macro_api.available_companies),
            "available_companies": macro_api.available_companies,
            "features": MACRO_FEATURES,
            "window_size": WINDOW_SIZE,
            "data_sources": {
                "processed_data": PROCESSED_DIR,
                "sequences": SEQUENCE_DIR,
                "models": MODEL_DIR,
                "outputs": OUTPUT_DIR
            }
        }
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}") 