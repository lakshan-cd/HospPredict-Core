from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

import torch
import numpy as np

from src.inference_utils import load_rf_model, load_scaler
from src.data_utils import load_features, get_wide_features

import os

# --- Paths ---
FEATURES_CSV = os.path.join("data", "feature_wide", "features_wide.csv")
RF_PATH = os.path.join("models", "rf_gnn_ensemble.joblib")
SCALER_PATH = os.path.join("models", "scaler", "tabular_scaler.joblib")

# --- Load at module import ---
features_df, feat_cols = load_features(FEATURES_CSV)
rf = load_rf_model(RF_PATH)
scaler = load_scaler(SCALER_PATH)

router = APIRouter()

class PredictRequest(BaseModel):
    company_id: str
    period: str  # 'YYYY-MM-DD'
    occupancy_drop: Optional[float] = 0.0

class PredictResponse(BaseModel):
    risk: float
    risk_perturbed: Optional[float] = None
    delta_risk: Optional[float] = None

@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # 1. Get features
    wide = get_wide_features(features_df, feat_cols, req.company_id, req.period)
    if wide is None:
        raise HTTPException(status_code=404, detail="Features not found for this company/period.")

    # 2. Scale features
    X = scaler.transform([wide])

    # 3. Predict risk
    prob = rf.predict_proba(X)[0, 1]
    result = {"risk": float(prob)}

    # 4. Hypothetical: Occupancy drop
    if req.occupancy_drop and req.occupancy_drop > 0:
        try:
            occ_idx = feat_cols.index('OccupancyRate')
        except ValueError:
            raise HTTPException(status_code=500, detail="OccupancyRate not found in features.")
        wide_perturbed = wide.copy()
        wide_perturbed[occ_idx] *= (1 - req.occupancy_drop / 100.0)
        X_perturbed = scaler.transform([wide_perturbed])
        prob_perturbed = rf.predict_proba(X_perturbed)[0, 1]
        result["risk_perturbed"] = float(prob_perturbed)
        result["delta_risk"] = float(prob_perturbed - prob)

    return result


@router.get("/list_companies")
def list_companies():
    return {"companies": features_df['company_id'].unique().tolist()}

@router.get("/list_periods/{company_id}")
def list_periods(company_id: str):
    periods = features_df[features_df['company_id'] == company_id]['period'].dt.strftime('%Y-%m-%d').unique().tolist()
    return {"periods": periods}