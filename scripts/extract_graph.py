# #!/usr/bin/env python3
# import os
# import yaml
# import torch
# from neo4j import GraphDatabase
# from torch_geometric.data import HeteroData

# def load_config():
#     cfg_path = os.path.join(os.path.dirname(__file__),
#                             '..','config','default.yml')
#     return yaml.safe_load(open(cfg_path))

# def get_driver(cfg):
#     return GraphDatabase.driver(cfg['neo4j']['uri'],
#                                 auth=(cfg['neo4j']['user'],
#                                       cfg['neo4j']['password']))

# def extract_for_company(driver, company_id, out_dir):
#     """
#     Pull all nodes & edges for one company and build a HeteroData.
#     """
#     data = HeteroData()
#     with driver.session() as sess:
#         # --- 1) FinancialMetric nodes & features ---
#         fm_query = """
#         MATCH (h:Hotel {company_id:$company})
#           -[:HAS_FINANCIAL_METRIC]->(fm:FinancialMetric)
#         RETURN id(fm) AS nid, fm.name AS name, fm.value AS val
#         """
#         fm_rows = sess.run(fm_query, {'company': company_id})
#         fm_nids, fm_names, fm_vals = [], [], []
#         for r in fm_rows:
#             fm_nids.append(r['nid'])
#             fm_names.append(r['name'])
#             fm_vals .append(0.0 if r['val'] is None else r['val'])
#         # assign an idx mapping
#         fm2idx = {nid:i for i,nid in enumerate(fm_nids)}
#         data['FinancialMetric'].x = torch.tensor(
#             fm_vals, dtype=torch.float).unsqueeze(1)  # shape [N_fm,1]

#         # store names as metadata if you like
#         data['FinancialMetric'].names = fm_names

#         # --- 2) StockMetric nodes & features ---
#         sm_query = """
#         MATCH (h:Hotel {company_id:$company})
#           -[:HAS_STOCK_PERFORMANCE]->(sm:StockMetric)
#         RETURN id(sm) AS nid, sm.volatility_30 AS v30, sm.daily_return AS dr
#         """
#         sm_rows = sess.run(sm_query, {'company': company_id})
#         sm_nids, v30s, drs = [], [], []
#         for r in sm_rows:
#             sm_nids.append(r['nid'])
#             v30s  .append(0.0 if r['v30']  is None else r['v30'])
#             drs   .append(0.0 if r['dr']   is None else r['dr'])
#         sm2idx = {nid:i for i,nid in enumerate(sm_nids)}
#         data['StockMetric'].x = torch.tensor(
#             list(zip(v30s, drs)), dtype=torch.float)  # [N_sm,2]

#         # --- 3) TimePeriod nodes (all periods used by either FinancialMetric OR StockMetric) ---
#         tp_query = """
#         MATCH (tp:TimePeriod)<-[:BELONGS_TO_PERIOD]-(n)
#         WHERE (n:FinancialMetric OR n:StockMetric)
#           AND n.company_id = $company
#         RETURN DISTINCT id(tp) AS nid, tp.period AS per
#         """
#         tp_rows = sess.run(tp_query, {'company': company_id})
#         tp_nids, periods = [], []
#         for r in tp_rows:
#             tp_nids.append(r['nid'])
#             periods.append(r['per'])
#         tp2idx = {nid:i for i,nid in enumerate(tp_nids)}
#         # We won’t assign features to TimePeriod (yet)
#         data['TimePeriod'].periods = periods

#         # --- 4) Build edge lists for each relation type ---
#         def build_edges(rel_type, src_label, dst_label, src2idx, dst2idx):
#             q = f"""
#             MATCH (n:{src_label})-[r:{rel_type}]->(m:{dst_label})
#             WHERE (n.company_id = $company) OR (m.company_id = $company)
#             RETURN id(n) AS sid, id(m) AS tid, r.weight AS w
#             """
#             rows = sess.run(q, {'company': company_id})
#             edge_list, edge_w = [], []
#             for r in rows:
#                 s,t = r['sid'], r['tid']
#                 if s in src2idx and t in dst2idx:
#                     edge_list.append([src2idx[s], dst2idx[t]])
#                     edge_w.append(0.0 if r['w'] is None else r['w'])
#             if edge_list:
#                 edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
#                 edge_weight= torch.tensor(edge_w, dtype=torch.float)
#                 data[src_label, rel_type, dst_label].edge_index  = edge_index
#                 data[src_label, rel_type, dst_label].edge_weight = edge_weight

#         # a) QoQ_CHANGE: FinancialMetric → FinancialMetric
#         build_edges('QoQ_CHANGE','FinancialMetric','FinancialMetric',
#                     fm2idx, fm2idx)
#         # b) HAS_VOLATILITY: TimePeriod → IndicatorSummary (we skip IndicatorSummary for now)
#         # c) BELONGS_TO_PERIOD: FinancialMetric → TimePeriod
#         build_edges('BELONGS_TO_PERIOD','FinancialMetric','TimePeriod',
#                     fm2idx, tp2idx)
#         # d) CRITICAL_PERIOD: TimePeriod → FinancialMetric
#         build_edges('CRITICAL_PERIOD','TimePeriod','FinancialMetric',
#                     tp2idx, fm2idx)
#         # e) CRITICAL_DAY: TimePeriod → StockMetric
#         build_edges('CRITICAL_DAY','TimePeriod','StockMetric',
#                     tp2idx, sm2idx)

#         # …add other relations you care about…

#     # Save the HeteroData
#     os.makedirs(out_dir, exist_ok=True)
#     path = os.path.join(out_dir, f'{company_id}.pt')
#     torch.save(data, path)
#     print(f"Saved graph for {company_id} → {path}")

# if __name__ == '__main__':
#     cfg    = load_config()
#     driver = get_driver(cfg)
#     out_d  = cfg['data']['processed']['graph_data']
#     # load all company IDs
#     with driver.session() as sess:
#         comps = [r['company_id'] for r
#                  in sess.run("MATCH (h:Hotel) RETURN h.company_id AS company_id")]
#     for c in comps:
#         extract_for_company(driver, c, out_d)
#     driver.close()


#!/usr/bin/env python3
import os
import yaml
import torch
import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from torch_geometric.data import HeteroData
from dotenv import load_dotenv

load_dotenv()

# ——— Load config ———
cfg_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'config', 'default.yml'
))
print("Using config:", cfg_path)
cfg = yaml.safe_load(open(cfg_path))

# ——— Neo4j driver ———
driver = GraphDatabase.driver(
    os.getenv('NEO4J_URI'),
    auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))
)

# ——— Load labels.csv ———
labels_fp = os.path.abspath(cfg['data']['labels']['out_csv'])
print("Loading labels from:", labels_fp)
labels_df = pd.read_csv(labels_fp, parse_dates=['period'])
label_map = {
    (row.company_id, row.period.date()): row.target_flag
    for row in labels_df.itertuples(index=False)
}
print(f"Loaded {len(label_map)} labels")

# ——— Load wide (tabular) features ———
feats_fp = os.path.abspath(cfg['data']['features']['wide_csv'])
print("Loading graph-level features from:", feats_fp)
feats_df = pd.read_csv(feats_fp, parse_dates=['period'])
feats_df['id'] = feats_df.company_id + '_' + feats_df.period.dt.strftime('%Y-%m-%d')
meta_cols = ['company_id','period','id']
feat_cols = [c for c in feats_df.columns if c not in meta_cols]

feat_map = {}
for row in feats_df.itertuples(index=False):
    arr = np.array([getattr(row, c) for c in feat_cols], dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    feat_map[row.id] = arr
num_feats = len(feat_cols)
print(f"Loaded {len(feat_map)} graph‐feature vectors (dim={num_feats})")

# ——— Prepare output directory ———
out_dir = os.path.abspath(cfg['data']['processed']['quarter_graphs'])
print("Writing quarter‐graphs into:", out_dir)
os.makedirs(out_dir, exist_ok=True)

def extract_quarter_graph(driver, company_id):
    # 1) Fetch all TimePeriod end_dates & internal IDs
    raw = []
    with driver.session() as sess:
        result = sess.run(
            """
            MATCH (h:Hotel {company_id:$c})
              -[:HAS_FINANCIAL_METRIC]->(:FinancialMetric)
              -[:BELONGS_TO_PERIOD]->(tp:TimePeriod)
            RETURN DISTINCT tp.end_date AS end_date, id(tp) AS tp_id
            """, {'c': company_id}
        )
        for rec in result:
            dt = rec['end_date']
            # neo4j Date ➔ python date
            d = dt.to_native() if hasattr(dt, 'to_native') else dt
            raw.append((d, rec['tp_id']))

    raw.sort(key=lambda x: x[0])
    if not raw:
        print(f"\nCompany {company_id}: no periods found, skipping")
        return
    periods, tp_ids = zip(*raw)
    print(f"\nCompany {company_id}: {len(periods)} periods: {periods}")

    saved = 0
    # 2) For each quarter except the last
    for i in range(len(periods) - 1):
        per, next_p = periods[i], periods[i+1]
        key = (company_id, next_p)
        if key not in label_map:
            print(f"  • {per} → {next_p}: no label, skip")
            continue
        flag = label_map[key]

        data = HeteroData()

        # ——— A) GraphFeature node ———
        gid = f"{company_id}_{per.isoformat()}"
        if gid not in feat_map:
            print(f"  ⚠️  No wide‐feature for {gid}, using zeros")
            gf = torch.zeros(num_feats, dtype=torch.float)
        else:
            gf = torch.tensor(feat_map[gid], dtype=torch.float)
        data['GraphFeature'].x = gf.unsqueeze(0)  # shape [1, num_feats]

        # ——— B) FinancialMetric nodes ———
        fm2idx, vals, names = {}, [], []
        with driver.session() as sess:
            rows = sess.run(
                """
                MATCH (h:Hotel {company_id:$c})
                  -[:HAS_FINANCIAL_METRIC]->(fm)-[:BELONGS_TO_PERIOD]->(tp)
                WHERE tp.end_date <= date($per)
                RETURN id(fm) AS fid, fm.name AS name, fm.value AS val
                """, {'c': company_id, 'per': per.isoformat()}
            )
            for idx, r in enumerate(rows):
                fm2idx[r['fid']] = idx
                names.append(r['name'])
                vals.append(0.0 if r['val'] is None else r['val'])
        if vals:
            data['FinancialMetric'].x     = torch.tensor(vals).unsqueeze(1)
            data['FinancialMetric'].names = names
        else:
            data['FinancialMetric'].x = torch.empty((0,1), dtype=torch.float)

        # ——— C) StockMetric nodes ———
        sm2idx, feats = {}, []
        with driver.session() as sess:
            rows = sess.run(
                """
                MATCH (h:Hotel {company_id:$c})
                  -[:HAS_STOCK_PERFORMANCE]->(sm)-[:BELONGS_TO_PERIOD]->(tp)
                WHERE tp.end_date <= date($per)
                RETURN id(sm) AS sid, sm.volatility_30 AS v30, sm.daily_return AS dr
                """, {'c': company_id, 'per': per.isoformat()}
            )
            for idx, r in enumerate(rows):
                sm2idx[r['sid']] = idx
                feats.append([
                    0.0 if r['v30'] is None else r['v30'],
                    0.0 if r['dr'] is None else r['dr']
                ])
        if feats:
            data['StockMetric'].x = torch.tensor(feats, dtype=torch.float)
        else:
            data['StockMetric'].x = torch.empty((0,2), dtype=torch.float)

        # ——— D) TimePeriod nodes ———
        tp2idx = {nid: j for j, nid in enumerate(tp_ids[:i+1])}
        data['TimePeriod'].periods = list(periods[:i+1])

        # ——— E) Build edges ———
        def build(rel, src, dst, s_map, d_map):
            edges, weights = [], []
            with driver.session() as sess:
                rows = sess.run(
                    f"""
                    MATCH (a:{src})-[r:{rel}]->(b:{dst})
                    WHERE a.company_id=$c AND id(a) IN $sa AND id(b) IN $sb
                    RETURN id(a) AS a_, id(b) AS b_, r.weight AS w
                    """, {
                        'c': company_id,
                        'sa': list(s_map.keys()),
                        'sb': list(d_map.keys())
                    }
                )
                for r in rows:
                    edges.append([s_map[r['a_']], d_map[r['b_']]])
                    weights.append(0.0 if r['w'] is None else r['w'])
            if edges:
                ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
                data[src, rel, dst].edge_index  = ei
                data[src, rel, dst].edge_weight = torch.tensor(weights, dtype=torch.float)

        build('QoQ_CHANGE',        'FinancialMetric','FinancialMetric', fm2idx, fm2idx)
        build('BELONGS_TO_PERIOD', 'FinancialMetric','TimePeriod',      fm2idx, tp2idx)
        build('CRITICAL_PERIOD',   'TimePeriod',      'FinancialMetric', tp2idx, fm2idx)
        build('CRITICAL_DAY',      'TimePeriod',      'StockMetric',      tp2idx, sm2idx)

        # ——— Save snapshot ———
        fname = f"{company_id}_{per.isoformat()}.pt"
        torch.save({'graph': data, 'label': flag},
                   os.path.join(out_dir, fname))
        print(f"  ✅ Saved {fname} (label={flag})")
        saved += 1

    print(f"→ {saved} graphs written for {company_id}")

if __name__ == '__main__':
    with driver.session() as sess:
        comps = [r['company_id']
                 for r in sess.run("MATCH (h:Hotel) RETURN h.company_id AS company_id")]
    print(f"Will process {len(comps)} companies: {comps}")
    for comp in comps:
        extract_quarter_graph(driver, comp)
    driver.close()
    print("✅ All done.")









