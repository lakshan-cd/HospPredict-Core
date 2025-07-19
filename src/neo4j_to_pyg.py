import os
import torch
import numpy as np
from neo4j import GraphDatabase
from torch_geometric.data import HeteroData
import pandas as pd

# --- Load wide features (for GraphFeature node) ---
def load_feat_map(features_csv):
    feats_df = pd.read_csv(features_csv, parse_dates=['period'])
    feats_df['id'] = feats_df.company_id + '_' + feats_df.period.dt.strftime('%Y-%m-%d')
    meta_cols = ['company_id','period','id']
    feat_cols = [c for c in feats_df.columns if c not in meta_cols]
    feat_map = {}
    for row in feats_df.itertuples(index=False):
        arr = np.array([getattr(row, c) for c in feat_cols], dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        feat_map[row.id] = arr
    return feat_map, feat_cols

def fetch_hotel_graph(driver, company_id, period, feat_map, feat_cols):
    """
    Fetches a HeteroData graph for a given company and period from Neo4j.
    - company_id: e.g. "Aitken_Spence_Hotel_Holdings_PLC"
    - period: e.g. "2014-03-31" (string, ISO format)
    - feat_map: dict from id to wide feature np.array
    - feat_cols: list of feature column names
    """
    data = HeteroData()
    with driver.session() as sess:
        # --- FinancialMetric nodes & features ---
        fm_query = """
        MATCH (h:Hotel {company_id:$company})
          -[:HAS_FINANCIAL_METRIC]->(fm:FinancialMetric)
        RETURN id(fm) AS nid, fm.name AS name, fm.value AS val
        """
        fm_rows = sess.run(fm_query, {'company': company_id})
        fm_nids, fm_names, fm_vals = [], [], []
        for r in fm_rows:
            fm_nids.append(r['nid'])
            fm_names.append(r['name'])
            fm_vals .append(0.0 if r['val'] is None else r['val'])
        fm2idx = {nid:i for i,nid in enumerate(fm_nids)}
        data['FinancialMetric'].x = torch.tensor(fm_vals, dtype=torch.float).unsqueeze(1) if fm_vals else torch.empty((0,1), dtype=torch.float)
        data['FinancialMetric'].names = fm_names

        # --- StockMetric nodes & features ---
        sm_query = """
        MATCH (h:Hotel {company_id:$company})
          -[:HAS_STOCK_PERFORMANCE]->(sm:StockMetric)
        RETURN id(sm) AS nid, sm.volatility_30 AS v30, sm.daily_return AS dr
        """
        sm_rows = sess.run(sm_query, {'company': company_id})
        sm_nids, v30s, drs = [], [], []
        for r in sm_rows:
            sm_nids.append(r['nid'])
            v30s  .append(0.0 if r['v30']  is None else r['v30'])
            drs   .append(0.0 if r['dr']   is None else r['dr'])
        sm2idx = {nid:i for i,nid in enumerate(sm_nids)}
        data['StockMetric'].x = torch.tensor(list(zip(v30s, drs)), dtype=torch.float) if v30s else torch.empty((0,2), dtype=torch.float)

        # --- TimePeriod nodes ---
        tp_query = """
        MATCH (tp:TimePeriod)<-[:BELONGS_TO_PERIOD]-(n)
        WHERE (n:FinancialMetric OR n:StockMetric)
          AND n.company_id = $company
        RETURN DISTINCT id(tp) AS nid, tp.period AS per
        """
        tp_rows = sess.run(tp_query, {'company': company_id})
        tp_nids, periods = [], []
        for r in tp_rows:
            tp_nids.append(r['nid'])
            periods.append(r['per'])
        tp2idx = {nid:i for i,nid in enumerate(tp_nids)}
        data['TimePeriod'].periods = periods
        data['TimePeriod'].num_nodes = len(periods)

        # --- GraphFeature node (wide features) ---
        gid = f"{company_id}_{period}"
        if gid in feat_map:
            gf = torch.tensor(feat_map[gid], dtype=torch.float)
        else:
            gf = torch.zeros(len(feat_cols), dtype=torch.float)
        data['GraphFeature'].x = gf.unsqueeze(0)

        # --- Build edge lists for each relation type ---
        def build_edges(rel_type, src_label, dst_label, src2idx, dst2idx):
            q = f"""
            MATCH (n:{src_label})-[r:{rel_type}]->(m:{dst_label})
            WHERE (n.company_id = $company) OR (m.company_id = $company)
            RETURN id(n) AS sid, id(m) AS tid, r.weight AS w
            """
            rows = sess.run(q, {'company': company_id})
            edge_list, edge_w = [], []
            for r in rows:
                s,t = r['sid'], r['tid']
                if s in src2idx and t in dst2idx:
                    edge_list.append([src2idx[s], dst2idx[t]])
                    edge_w.append(0.0 if r['w'] is None else r['w'])
            if edge_list:
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                edge_weight= torch.tensor(edge_w, dtype=torch.float)
                data[src_label, rel_type, dst_label].edge_index  = edge_index
                data[src_label, rel_type, dst_label].edge_weight = edge_weight

        build_edges('QoQ_CHANGE','FinancialMetric','FinancialMetric', fm2idx, fm2idx)
        build_edges('BELONGS_TO_PERIOD','FinancialMetric','TimePeriod', fm2idx, tp2idx)
        build_edges('CRITICAL_PERIOD','TimePeriod','FinancialMetric', tp2idx, fm2idx)
        build_edges('CRITICAL_DAY','TimePeriod','StockMetric', tp2idx, sm2idx)

    return data