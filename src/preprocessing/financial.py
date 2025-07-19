import pandas as pd
import numpy as np

def preprocess_financial(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and prepares financial report data, computing additional ratios 
    and flags for FTA-GNN. If ROA/ROE are already present, only fills missing.
    """
    df = df.copy()

    # 1) Parse datetime columns
    for col in ['PeriodStart', 'PeriodEnd', 'PublishedDate']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # 2) Convert remaining columns to numeric (except identifiers)
    exclude = [
        'FiscalPeriod', 'SourceDocument', 'CompanyID', 'StatementType',
        'PresentationCurrency', 'AuditStatus', 'PeriodStart',
        'PeriodEnd', 'PublishedDate'
    ]
    for col in df.columns.difference(exclude):
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3) Fill numeric NaNs with column median
    for col in df.select_dtypes(include='number').columns:
        df[col].fillna(df[col].median(), inplace=True)

    # 4) Compute Working Capital
    if 'CurrentAssets' in df.columns and 'CurrentLiabilities' in df.columns:
        df['WorkingCapital'] = df['CurrentAssets'] - df['CurrentLiabilities']

    # 5) Basic financial ratios
    df['DebtToEquity']     = df['TotalLiabilities'] / df['ShareholdersEquity']
    df['CurrentRatio']     = df['CurrentAssets']    / df['CurrentLiabilities']
    df['QuickRatio']       = (df['CurrentAssets'] - df['Inventory']) / df['CurrentLiabilities']
    df['GrossMargin']      = df['GrossProfit']       / df['Revenue']
    df['NetProfitMargin']  = df['NetIncome']         / df['Revenue']

    # 6) ROA and ROE: only compute if absent, otherwise fill missing values
    roa_num = df['NetIncome']
    roa_den = df['TotalAssets']
    roe_den = df['ShareholdersEquity']

    if 'ROA' not in df.columns:
        df['ROA'] = roa_num / roa_den
    else:
        df['ROA'] = df['ROA'].fillna(roa_num / roa_den)

    if 'ROE' not in df.columns:
        df['ROE'] = roa_num / roe_den
    else:
        df['ROE'] = df['ROE'].fillna(roa_num / roe_den)

    # 7) (Optional) Altman Z-score if you have all required fields:
    # if set(df.columns) >= {'WorkingCapital','RetainedEarnings','EBITDA','MarketValueEquity','TotalLiabilities','Revenue','TotalAssets'}:
    #     df['AltmanZ'] = (
    #         1.2 * df['WorkingCapital'] / df['TotalAssets']
    #         + 1.4 * df['RetainedEarnings'] / df['TotalAssets']
    #         + 3.3 * df['EBITDA'] / df['TotalAssets']
    #         + 0.6 * df['MarketValueEquity'] / df['TotalLiabilities']
    #         + df['Revenue'] / df['TotalAssets']
    #     )

    # 8) Safe EBITDA jump / critical_quarter flag
    if 'EBITDA' in df.columns and df['EBITDA'].notna().any():
        # ensure proper ordering for diff
        df = df.sort_values(['CompanyID', 'PeriodEnd'])

        # QoQ difference
        df['EBITDA_diff'] = df.groupby('CompanyID')['EBITDA'].diff()

        # helper to avoid divide-by-zero
        def safe_z(series: pd.Series) -> pd.Series:
            sigma = series.std()
            if pd.isna(sigma) or sigma == 0:
                return pd.Series(0, index=series.index)
            return (series - series.mean()) / sigma

        df['EBITDA_z'] = df.groupby('CompanyID')['EBITDA_diff'].transform(safe_z)
        df['critical_quarter'] = (df['EBITDA_z'].abs() > 1).astype(int)
    else:
        # fill defaults if no EBITDA data
        df['EBITDA_diff']      = np.nan
        df['EBITDA_z']         = 0
        df['critical_quarter'] = 0

    return df
