#!/usr/bin/env python3
import os, glob, yaml, pandas as pd

def load_config():
    fn = os.path.join(os.path.dirname(__file__), '..', 'config', 'default.yml')
    return yaml.safe_load(open(fn))

def main():
    cfg     = load_config()
    src_dir = cfg['data']['labels']['source_dir']
    out_csv = cfg['data']['labels']['out_csv']
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    rows = []
    for fp in glob.glob(os.path.join(src_dir, '*_merged.csv')):
        comp = os.path.basename(fp).replace('_merged.csv','')
        df   = pd.read_csv(fp, parse_dates=['trade_date','PeriodEnd'])
        if df.empty:
            continue

        # 1) average 30-day volatility per quarter
        qv = df.groupby('PeriodEnd')['volatility_30'].mean().sort_index()
        if len(qv) < 2:
            continue

        # 2) next-quarter series and 75th-percentile flag
        next_qv = qv.shift(-1).dropna()
        median = next_qv.quantile(0.50)
        flags  = (next_qv >= median).astype(int)

        # 3) **emit one label row for _every_ quarter in next_qv**
        for per, flag in flags.items():
            rows.append({
                'company_id':  comp,
                'period':       per,
                'target_vol':   float(next_qv.loc[per]),
                'target_flag':  int(flag)
            })

    labels = pd.DataFrame(rows)
    labels.to_csv(out_csv, index=False)
    print(f"✅ Wrote {len(labels)} rows to {out_csv}")

if __name__ == "__main__":
    main()
