import pandas as pd

def preprocess_trade(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and prepares daily trade data.
    """
    df = df.copy()
    df.columns = (df.columns
                    .str.strip()
                    .str.lower()
                    .str.replace(' ', '_', regex=False)
                    .str.replace(r'[^\w_]', '', regex=True))
    df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
    for col in df.columns.drop('trade_date'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.sort_values('trade_date', inplace=True)
    df.ffill(inplace=True)
    df['daily_return'] = df['close_rs'].pct_change()
    df['return_ma7'] = df['daily_return'].rolling(7).mean()
    df['return_ma30'] = df['daily_return'].rolling(30).mean()
    df['volatility_30'] = df['daily_return'].rolling(30).std()

    # --- new features for volatility-driven attention ---
    # z-score of share volume so spikes are detectable:
    df['vol_zscore']     = (
         df['sharevolume'] - df['sharevolume'].mean()
      ) / df['sharevolume'].std()
    df['vol_zscore']     = df['vol_zscore'].fillna(0)
    # flag critical days where volatility_30 > μ+σ
    v30 = df['volatility_30']
    df['vol_thresh']     = ((v30 - v30.mean())/v30.std() > 1).astype(int)
    return df
