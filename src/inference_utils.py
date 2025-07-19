import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import joblib
from torch_geometric.nn import HeteroConv, GATConv
from torch_geometric.data import HeteroData

# ---- Model Classes ----

class GlobalTemporalAttention(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.query = nn.Linear(in_dim, hidden)
        self.key   = nn.Linear(in_dim, hidden)
        self.value = nn.Linear(in_dim, hidden)
        self.scale = hidden ** -0.5
    def forward(self, period_emb, period_vol):
        x = period_emb * period_vol  # [T, in_dim]
        Q, K, V = self.query(x), self.key(x), self.value(x)
        scores = torch.softmax(Q @ K.transpose(-1, -2) * self.scale, dim=-1)
        context = scores @ V
        global_vec = context.mean(dim=0)
        return global_vec, scores.detach()

class LocalIndicatorAttention(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.query = nn.Linear(in_dim, hidden)
        self.key   = nn.Linear(in_dim, hidden)
        self.value = nn.Linear(in_dim, hidden)
        self.scale = hidden ** -0.5
    def forward(self, fm_emb):
        if fm_emb.numel() == 0:
            hidden = self.query.weight.shape[0]
            return torch.zeros(hidden, device=fm_emb.device), torch.zeros((0,0), device=fm_emb.device)
        Q, K, V = self.query(fm_emb), self.key(fm_emb), self.value(fm_emb)
        scores = torch.softmax(Q @ K.transpose(-1, -2) * self.scale, dim=-1)
        context = scores @ V
        local_vec = context.mean(dim=0)
        return local_vec, scores.detach()

class HeteroFTAGNN_v3(nn.Module):
    def __init__(self, in_fm, in_sm, in_tp, in_gf, hidden=128, heads=2, dropout=0.2):
        super().__init__()
        self.hidden = hidden
        rel_convs = {
            ('FinancialMetric','QoQ_CHANGE','FinancialMetric'):
                GATConv(in_fm, hidden//heads, heads=heads, dropout=dropout, add_self_loops=False),
            ('FinancialMetric','BELONGS_TO_PERIOD','TimePeriod'):
                GATConv((in_fm,in_tp), hidden//heads, heads=heads, dropout=dropout, add_self_loops=False),
            ('TimePeriod','CRITICAL_PERIOD','FinancialMetric'):
                GATConv((in_tp,in_fm), hidden//heads, heads=heads, dropout=dropout, add_self_loops=False),
            ('TimePeriod','CRITICAL_DAY','StockMetric'):
                GATConv((in_tp,in_sm), hidden//heads, heads=heads, dropout=dropout, add_self_loops=False),
            ('TimePeriod','rev_BELONGS','FinancialMetric'):
                GATConv((in_tp,in_fm), hidden//heads, heads=heads, dropout=dropout, add_self_loops=False),
        }
        self.rel_keys = list(rel_convs.keys())
        self.conv1 = HeteroConv(rel_convs, aggr='sum')
        self.conv2 = HeteroConv({
            etype: GATConv(hidden, hidden//heads, heads=heads,
                           dropout=dropout, add_self_loops=False)
            for etype in self.rel_keys
        }, aggr='sum')
        self.ln = nn.ModuleDict({
            'FinancialMetric': nn.LayerNorm(hidden),
            'TimePeriod'     : nn.LayerNorm(hidden),
            'StockMetric'    : nn.LayerNorm(hidden),
        })
        self.global_att = GlobalTemporalAttention(hidden,hidden)
        self.local_att  = LocalIndicatorAttention(hidden,hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden*3 + in_gf, hidden//2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden//2,1)
        )

    def forward(self, data, period_vol, period_emb):
        h = self._embed(data, period_vol, period_emb)
        return self.head(h).squeeze(), None, None

    def _embed(self, data, period_vol, period_emb):
        T,dev = data['TimePeriod'].num_nodes, data['FinancialMetric'].x.device
        pe = period_emb(torch.arange(T,device=dev)); pe=torch.nan_to_num(pe)
        x = {
            'FinancialMetric': data['FinancialMetric'].x,
            'TimePeriod'     : pe,
            'StockMetric'    : data['StockMetric'].x,
        }
        def ei(rel):
            return data[rel].edge_index if rel in data.edge_types else torch.zeros((2,0),dtype=torch.long,device=dev)
        edges = {etype: ei(etype) for etype in self.rel_keys}
        out1 = self.conv1(x, edges)
        fm1 = F.elu(self.ln['FinancialMetric'](out1['FinancialMetric']))
        tp1 = F.elu(self.ln['TimePeriod'](out1['TimePeriod']))
        sm1 = F.elu(self.ln['StockMetric'](out1['StockMetric']))
        out2 = self.conv2({'FinancialMetric':fm1,'TimePeriod':tp1,'StockMetric':sm1}, edges)
        fm2 = F.elu(self.ln['FinancialMetric'](out2['FinancialMetric'] + fm1))
        tp2 = F.elu(self.ln['TimePeriod'](out2['TimePeriod']    + tp1))
        sm2 = F.elu(self.ln['StockMetric'](out2['StockMetric']  + sm1))
        global_vec,_ = self.global_att(tp2,period_vol)
        local_vec,_  = self.local_att(fm2)
        sm_pool      = sm2.mean(dim=0) if sm2.size(0)>0 else torch.zeros(self.hidden,device=dev)
        gf           = data['GraphFeature'].x.squeeze(0)
        return torch.cat([local_vec,global_vec,sm_pool,gf],dim=-1)

# ---- Helper Functions ----

def compute_period_vol(g):
    rel = ('TimePeriod','CRITICAL_DAY','StockMetric')
    T = g['TimePeriod'].num_nodes
    if rel not in g.edge_types or not hasattr(g[rel], 'edge_index'):
        return torch.zeros((T,1), device=g['TimePeriod'].periods_emb.device)
    v30 = g['StockMetric'].x[:,0]
    ei = g[rel].edge_index; p_i, sm_i = ei[0], ei[1]
    vols = [(v30[sm_i[p_i==t]].mean() if (p_i==t).any() else torch.tensor(0.,device=v30.device))
            for t in range(T)]
    return torch.stack(vols).unsqueeze(1)

# ---- Feature Preprocessing ----

def preprocess_features(features_df):
    # This matches your training pipeline
    features_df['id'] = features_df.company_id + '_' + pd.to_datetime(features_df.period).dt.strftime('%Y-%m-%d')
    feat_cols = [c for c in features_df.columns if c not in ('company_id','period','id')]
    low, high = features_df[feat_cols].quantile(0.01), features_df[feat_cols].quantile(0.99)
    features_df[feat_cols] = features_df[feat_cols].clip(low, high, axis=1)
    return features_df, feat_cols

def load_wide_features(features_csv):
    features_df = pd.read_csv(features_csv, parse_dates=['period'])
    features_df, feat_cols = preprocess_features(features_df)
    return features_df, feat_cols

# ---- Model/Scaler Loaders ----

def load_gnn_model(model_path, in_fm, in_sm, in_tp, in_gf, device='cpu'):
    model = HeteroFTAGNN_v3(in_fm, in_sm, in_tp, in_gf)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def load_rf_model(rf_path):
    return joblib.load(rf_path)

def load_scaler(scaler_path):
    return joblib.load(scaler_path)