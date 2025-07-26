from fastapi import APIRouter, HTTPException, Query
import pandas as pd
from typing import List, Optional
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
import numpy as np

router = APIRouter()

@router.get("/companies")
def get_companies():
    df = pd.read_csv("data/feature_wide/features_wide.csv")
    companies = df["company_id"].unique().tolist()
    return {"companies": [{"id": cid, "name": cid} for cid in companies]}

@router.get("/company/{company_id}/metrics")
def get_company_metrics(company_id: str):
    df = pd.read_csv("data/feature_wide/features_wide.csv")
    if company_id not in df["company_id"].unique():
        raise HTTPException(status_code=404, detail="Company not found")
    exclude = {"company_id", "period"}
    metrics = [col for col in df.columns if col not in exclude]
    return {"metrics": metrics}

@router.get("/company/{company_id}/trend/{metric}")
def get_company_trend(company_id: str, metric: str):
    df = pd.read_csv("data/feature_wide/features_wide.csv", parse_dates=["period"])
    company_df = df[df["company_id"] == company_id].sort_values("period")
    if company_df.empty:
        raise HTTPException(status_code=404, detail="Company not found or no data")
    if metric not in company_df.columns:
        raise HTTPException(status_code=404, detail="Metric not found")
    return {
        "company_id": company_id,
        "metric": metric,
        "data": [{"period": p.strftime("%Y-%m-%d"), "value": v} for p, v in zip(company_df["period"], company_df[metric])]
    }

@router.get("/company/{company_id}/trend/multi")
def get_company_trend_multi(company_id: str, metrics: List[str] = Query(...)):
    df = pd.read_csv("data/feature_wide/features_wide.csv", parse_dates=["period"])
    df.columns = df.columns.str.strip()  # Remove whitespace from columns
    company_df = df[df["company_id"] == company_id].sort_values("period")
    if company_df.empty:
        raise HTTPException(status_code=404, detail="Company not found or no data")
    metrics = [m.strip() for m in metrics]  # Remove whitespace from requested metrics
    for metric in metrics:
        if metric not in company_df.columns:
            raise HTTPException(status_code=404, detail=f"Metric '{metric}' not found")
    result = {}
    for metric in metrics:
        result[metric] = [{"period": p.strftime("%Y-%m-%d"), "value": v} for p, v in zip(company_df["period"], company_df[metric])]
    return {
        "company_id": company_id,
        "metrics": metrics,
        "data": result
    }

@router.get("/company/{company_id}/analytics/moving_average/{metric}")
def get_moving_average(company_id: str, metric: str, window: int = 3):
    df = pd.read_csv("data/feature_wide/features_wide.csv", parse_dates=["period"])
    company_df = df[df["company_id"] == company_id].sort_values("period")
    if company_df.empty:
        raise HTTPException(status_code=404, detail="Company not found or no data")
    if metric not in company_df.columns:
        raise HTTPException(status_code=404, detail="Metric not found")
    ma = company_df[metric].rolling(window=window, min_periods=1).mean()
    return {
        "company_id": company_id,
        "metric": metric,
        "window": window,
        "moving_average": [{"period": p.strftime("%Y-%m-%d"), "value": v} for p, v in zip(company_df["period"], ma)]
    }

@router.get("/company/{company_id}/analytics/seasonality/{metric}")
def get_seasonality_decomposition(company_id: str, metric: str, period: int = 4):
    df = pd.read_csv("data/feature_wide/features_wide.csv", parse_dates=["period"])
    company_df = df[df["company_id"] == company_id].sort_values("period")
    if company_df.empty:
        raise HTTPException(status_code=404, detail="Company not found or no data")
    if metric not in company_df.columns:
        raise HTTPException(status_code=404, detail="Metric not found")
    series = company_df[metric].values
    if len(series) < period * 2:
        raise HTTPException(status_code=400, detail="Not enough data for decomposition")
    try:
        result = seasonal_decompose(series, period=period, model='additive', extrapolate_trend='freq')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Decomposition failed: {str(e)}")
    periods = company_df["period"].dt.strftime("%Y-%m-%d").tolist()
    return {
        "company_id": company_id,
        "metric": metric,
        "periods": periods,
        "trend": result.trend.tolist(),
        "seasonal": result.seasonal.tolist(),
        "resid": result.resid.tolist(),
        "observed": result.observed.tolist()
    }

@router.get("/company/{company_id}/analytics/anomaly/{metric}")
def get_anomaly_detection(company_id: str, metric: str, z_thresh: float = 2.5):
    df = pd.read_csv("data/feature_wide/features_wide.csv", parse_dates=["period"])
    company_df = df[df["company_id"] == company_id].sort_values("period")
    if company_df.empty:
        raise HTTPException(status_code=404, detail="Company not found or no data")
    if metric not in company_df.columns:
        raise HTTPException(status_code=404, detail="Metric not found")
    values = company_df[metric].values
    z_scores = np.abs(stats.zscore(values, nan_policy='omit'))
    anomalies = [
        {"period": p.strftime("%Y-%m-%d"), "value": float(v)}
        for p, v, z in zip(company_df["period"], values, z_scores)
        if not np.isnan(z) and z > z_thresh
    ]
    return {
        "company_id": company_id,
        "metric": metric,
        "z_thresh": float(z_thresh),
        "anomalies": anomalies
    }
