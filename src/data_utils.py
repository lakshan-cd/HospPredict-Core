import pandas as pd
import numpy as np

def load_features(features_csv):
    df = pd.read_csv(features_csv, parse_dates=['period'])
    df['id'] = df['company_id'] + '_' + df['period'].dt.strftime('%Y-%m-%d')
    feat_cols = [c for c in df.columns if c not in ('company_id','period','id')]
    # Clipping and preprocessing as in training
    low, high = df[feat_cols].quantile(0.01), df[feat_cols].quantile(0.99)
    df[feat_cols] = df[feat_cols].clip(low, high, axis=1)
    return df, feat_cols

def get_wide_features(df, feat_cols, company_id, period):
    # period should be a datetime or string in 'YYYY-MM-DD'
    if not isinstance(period, str):
        period = pd.to_datetime(period).strftime('%Y-%m-%d')
    id_str = f"{company_id}_{period}"
    row = df[df['id'] == id_str]
    if row.empty:
        return None
    return row.iloc[0][feat_cols].values.astype(np.float32)

def load_labels(labels_csv):
    df = pd.read_csv(labels_csv, parse_dates=['period'])
    df['id'] = df['company_id'] + '_' + df['period'].dt.strftime('%Y-%m-%d')
    label_map = dict(zip(df['id'], df['target_flag']))
    return label_map