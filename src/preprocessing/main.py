import os
import pandas as pd
from glob import glob
import yaml

from src.preprocessing.financial import preprocess_financial
from src.preprocessing.trades import preprocess_trade

def main():
    cfg_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'default.yml')
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    fin_root = cfg['data']['raw']['financial']
    trade_root = cfg['data']['raw']['trades']
    processed_fin = cfg['data']['processed']['financial']
    processed_trade = cfg['data']['processed']['trades']
    merged_output = cfg['data']['processed']['merged']

    for d in [processed_fin, processed_trade, merged_output]:
        os.makedirs(d, exist_ok=True)

    companies = [os.path.basename(d) for d in glob(os.path.join(fin_root, '*')) if os.path.isdir(d)]
    for comp in companies:
        print(f'Processing {comp}...')
        fin_files = glob(os.path.join(fin_root, comp, '*.*'))
        fin_dfs = []
        for fp in fin_files:
            if fp.lower().endswith('.csv'):
                fin_dfs.append(pd.read_csv(fp))
            elif fp.lower().endswith(('.xls', '.xlsx')):
                fin_dfs.append(pd.read_excel(fp))
        if not fin_dfs:
            print(f'No financial data for {comp}')
            continue
        fin_df = pd.concat(fin_dfs, ignore_index=True)
        fin_clean = preprocess_financial(fin_df)
        fin_clean.to_csv(os.path.join(processed_fin, f'{comp}_financial_preprocessed.csv'), index=False)

        trade_files = glob(os.path.join(trade_root, comp, '*.*'))
        trade_dfs = []
        for fp in trade_files:
            if fp.lower().endswith('.csv'):
                trade_dfs.append(pd.read_csv(fp))
            elif fp.lower().endswith(('.xls', '.xlsx')):
                trade_dfs.append(pd.read_excel(fp))
        if not trade_dfs:
            print(f'No trade data for {comp}')
            continue
        trade_df = pd.concat(trade_dfs, ignore_index=True)
        trade_clean = preprocess_trade(trade_df)
        trade_clean.to_csv(os.path.join(processed_trade, f'{comp}_trade_preprocessed.csv'), index=False)

        merged = pd.merge_asof(
            trade_clean.sort_values('trade_date'),
            fin_clean.sort_values('PeriodEnd'),
            left_on='trade_date', right_on='PeriodEnd', direction='backward'
        )
        merged.to_csv(os.path.join(merged_output, f'{comp}_merged.csv'), index=False)

    print('ETL preprocessing completed.')

if __name__ == '__main__':
    main()
