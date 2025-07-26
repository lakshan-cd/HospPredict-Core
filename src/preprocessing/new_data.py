import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import RobustScaler
import yaml
from .financial import preprocess_financial
from .trades import preprocess_trade

def load_config():
    """Load configuration from default.yml"""
    with open("config/default.yml", "r") as f:
        return yaml.safe_load(f)

def preprocess_new_financial_data(financial_csv_path, company_id):
    """
    Preprocess new financial data for a company
    Returns: preprocessed DataFrame
    """
    # Load and preprocess using existing function
    df = pd.read_csv(financial_csv_path)
    df = preprocess_financial(df)
    
    # Save to processed/financial/
    output_path = f"data/processed/financial/{company_id}_financial_preprocessed.csv"
    df.to_csv(output_path, index=False)
    return df

def preprocess_new_trade_data(trade_csv_path, company_id):
    """
    Preprocess new trade data for a company
    Returns: preprocessed DataFrame
    """
    # Load and preprocess using existing function
    df = pd.read_csv(trade_csv_path)
    df = preprocess_trade(df)
    
    # Save to processed/trades/
    output_path = f"data/processed/trades/{company_id}_trade_preprocessed.csv"
    df.to_csv(output_path, index=False)
    return df

def merge_new_data(financial_df, trade_df, company_id):
    """
    Merge financial and trade data for a company
    Returns: merged DataFrame
    """
    # Apply same merging logic as in your existing pipeline
    # Use merge_asof for proper time-based merging
    merged_df = pd.merge_asof(
        trade_df.sort_values('trade_date'),
        financial_df.sort_values('PeriodEnd'),
        left_on='trade_date', right_on='PeriodEnd', direction='backward'
    )
    
    # Save to processed/merged/
    output_path = f"data/processed/merged/{company_id}_merged.csv"
    merged_df.to_csv(output_path, index=False)
    return merged_df

def generate_wide_features_for_new_data(merged_df, company_id, period):
    """
    Generate wide features for new data (single company/period)
    Returns: wide features DataFrame with one row
    """
    # Apply same logic as your feature_wide generation code
    if merged_df.empty:
        return None
    
    # a) last financial row per quarter
    fin = merged_df.set_index('PeriodEnd').sort_index().groupby('PeriodEnd').last()
    fin_num = fin.select_dtypes(include='number')
    
    # b) QoQ change for numeric financials
    qoq = fin_num.diff().add_suffix('_QoQ')
    
    # c) Stock aggregates per quarter
    stk = merged_df.groupby('PeriodEnd').agg({
        'volatility_30': 'mean',
        'daily_return': 'mean'
    }).rename(columns={'volatility_30': 'vol_mean', 'daily_return': 'ret_mean'})
    stk_qoq = stk.diff().add_suffix('_QoQ')
    
    # d) Combine, drop first row, fill NaNs, clamp inf→0
    wide = pd.concat([fin_num, qoq, stk, stk_qoq], axis=1)
    if len(wide) > 0:
        wide = wide.iloc[1:].copy()          # drop first-quarter NaNs
    wide = wide.fillna(0.0)                 # fill other NaNs
    wide = wide.replace([np.inf, -np.inf], 0.0)  # clamp infinities
    
    # Add company and period info (same as training pipeline)
    wide['company_id'] = company_id
    wide['period'] = wide.index
    wide = wide.reset_index(drop=True)
    
    return wide

def append_to_features_wide(wide_features_df):
    """
    Append new wide features to existing features_wide.csv
    """
    # Load existing features
    existing_features = pd.read_csv("data/feature_wide/features_wide.csv", parse_dates=['period'])
    
    # Append new features
    updated_features = pd.concat([existing_features, wide_features_df], ignore_index=True)
    
    # Save updated features
    updated_features.to_csv("data/feature_wide/features_wide.csv", index=False)
    
    return updated_features

def create_feature_map_for_new_data(wide_features_df):
    """
    Create feature map for new data (scaled features) - same as training pipeline
    Returns: feature map dict and feature columns list
    """
    # Get feature columns (same as training pipeline)
    feat_cols = [c for c in wide_features_df.columns if c not in ('company_id', 'period', 'id')]
    
    # Clip outliers (same as training pipeline)
    low, high = wide_features_df[feat_cols].quantile(0.01), wide_features_df[feat_cols].quantile(0.99)
    wide_features_df[feat_cols] = wide_features_df[feat_cols].clip(low, high, axis=1)
    
    # Scale features (same as training pipeline)
    wide_features_df[feat_cols] = RobustScaler().fit_transform(wide_features_df[feat_cols])
    
    # Create feature map (same as training pipeline)
    feat_map = {
        row.id: torch.tensor([getattr(row, c) for c in feat_cols], dtype=torch.float).unsqueeze(0)
        for row in wide_features_df[['id'] + feat_cols].itertuples(index=False)
    }
    
    return feat_map, feat_cols