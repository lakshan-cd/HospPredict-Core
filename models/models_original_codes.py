# ──────────── §0 Imports & Paths ─────────────
import os, glob, random
import torch, torch.nn.functional as F
import pandas as pd, numpy as np
from collections import Counter
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from torch_geometric.loader import DataLoader
from torch_geometric.nn import HeteroConv, GATConv, SAGEConv
from torch_geometric.data.storage import BaseStorage, NodeStorage, EdgeStorage
import torch.nn as nn
import datetime
from torch.serialization import add_safe_globals

# ──────────── §1 Repro & Device ─────────────
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ──────────── §2 Load & Scale Tabular Features ─────────────
FEATURES_CSV = '/content/drive/MyDrive/Research/financial_knowledge_graph/features_wide.csv'
gf_df = pd.read_csv(FEATURES_CSV, parse_dates=['period'])
gf_df['id'] = gf_df.company_id + '_' + gf_df.period.dt.strftime('%Y-%m-%d')
feat_cols = [c for c in gf_df.columns if c not in ('company_id','period','id')]
low, high = gf_df[feat_cols].quantile(0.01), gf_df[feat_cols].quantile(0.99)
gf_df[feat_cols] = gf_df[feat_cols].clip(low, high, axis=1)
gf_df[feat_cols] = RobustScaler().fit_transform(gf_df[feat_cols])
in_gf = len(feat_cols)
gf_map = {row.id: torch.tensor([getattr(row,c) for c in feat_cols],dtype=torch.float).unsqueeze(0)
          for row in gf_df[['id']+feat_cols].itertuples(index=False)}

# ──────────── §3 Load Graphs & Labels ─────────────
LABEL_CSV = '/content/drive/MyDrive/Research/financial_knowledge_graph/label/labels.csv'
labels_df = pd.read_csv(LABEL_CSV, parse_dates=['period'])
labels_df['id'] = labels_df.company_id + '_' + labels_df.period.dt.strftime('%Y-%m-%d')
label_map = dict(zip(labels_df.id, labels_df.target_flag))

# allow torch-geom to unpickle
torch.serialization.add_safe_globals([BaseStorage, NodeStorage, EdgeStorage])

# Allow datetime.date in the unpickler
add_safe_globals([datetime.date])


QG_DIR = '/content/drive/MyDrive/Research/financial_knowledge_graph/graph_data/quarter_data'
graphs, ids, flags = [], [], []
for fp in sorted(glob.glob(f"{QG_DIR}/*.pt")):
    bid = os.path.basename(fp).replace('.pt','')
    if bid in label_map and bid in gf_map:
        # explicitly tell torch.load to load the full object
        obj = torch.load(fp, weights_only=False)
        g = obj['graph']
        g['GraphFeature'].x = gf_map[bid]
        graphs.append(g)
        ids.append(bid)
        flags.append(int(label_map[bid]))

# ──────────── §4 FM/SM Scaling & Edge Sanitization ─────────────
# (same as your §3)
# ──────────── §3 FM/SM Feature‐Scaling & Edge‐Sanitization ─────────────
# 3.1 Convert to float
for g in graphs:
    g['FinancialMetric'].x = g['FinancialMetric'].x.float()
    g['StockMetric'].x     = g['StockMetric'].x.float()

# 3.2 Collect feature matrices
fm_list = [g['FinancialMetric'].x.cpu().numpy() for g in graphs if g['FinancialMetric'].x.numel()>0]
sm_list = [g['StockMetric'].x.cpu().numpy()    for g in graphs if g['StockMetric'].x.numel()>0]

# 3.3 Fit scalers
if fm_list:
    A = np.vstack(fm_list)
    A[~np.isfinite(A)] = np.nan
    A = A[~np.isnan(A).any(axis=1)]
    fm_scaler = StandardScaler().fit(A)
else:
    fm_scaler = None

if sm_list:
    B = np.vstack(sm_list)
    B[~np.isfinite(B)] = np.nan
    B = B[~np.isnan(B).any(axis=1)]
    sm_scaler = StandardScaler().fit(B)
else:
    sm_scaler = None

# 3.4 Transform back
for g in graphs:
    if fm_scaler and g['FinancialMetric'].x.numel()>0:
        arr = g['FinancialMetric'].x.cpu().double().numpy()
        arr[~np.isfinite(arr)] = 0.0
        g['FinancialMetric'].x = torch.from_numpy(fm_scaler.transform(arr)).float()
    if sm_scaler and g['StockMetric'].x.numel()>0:
        arr = g['StockMetric'].x.cpu().double().numpy()
        arr[~np.isfinite(arr)] = 0.0
        g['StockMetric'].x = torch.from_numpy(sm_scaler.transform(arr)).float()

# 3.5 Clean NaNs/Infs
for g in graphs:
    for key in ['FinancialMetric','StockMetric','GraphFeature']:
        if hasattr(g[key], 'x'):
            g[key].x = torch.nan_to_num(g[key].x, nan=0.0, posinf=0.0, neginf=0.0)

# 3.6 Set num_nodes and sanitize edges
for g in graphs:
    for nt in g.node_types:
        st = g[nt]
        if st.num_nodes is None:
            if hasattr(st, 'x'):
                st.num_nodes = st.x.size(0)
            elif hasattr(st, 'periods'):
                st.num_nodes = len(st.periods)
            else:
                st.num_nodes = 0
    for src, rel, dst in g.edge_types:
        es = g[src, rel, dst]
        if hasattr(es, 'edge_index'):
            ei = es.edge_index
            n1, n2 = g[src].num_nodes, g[dst].num_nodes
            mask = (ei[0] < n1) & (ei[1] < n2)
            es.edge_index = ei[:, mask]
            if hasattr(es, 'edge_weight'):
                es.edge_weight = es.edge_weight[mask]
print("✅ FM/SM scaling + edge sanitization done.")

# ──────────── §4 TimePeriod Embedding ─────────────
for g in graphs:
    g['TimePeriod'].num_nodes = len(g['TimePeriod'].periods)

num_T = max(len(g['TimePeriod'].periods) for g in graphs)
in_tp = 32
period_emb = nn.Embedding(num_T, in_tp).to(device)
for g in graphs:
    T = len(g['TimePeriod'].periods)
    g['TimePeriod'].periods_emb = period_emb(torch.arange(T, device=device))

# ──────────── §5 TimePeriod Embedding & compute_period_vol ─────────────
num_T = max(len(g['TimePeriod'].periods) for g in graphs)
in_tp = 32
period_emb = nn.Embedding(num_T, in_tp).to(device)

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

# ──────────── §5 Explicit Attention Modules (Corrected input dims) ─────────────
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

# ──────────── §6 HeteroSAGENet (mean aggregator) ─────────────
class HeteroSAGENet(nn.Module):
    def __init__(self, in_fm, in_sm, in_tp, in_gf, hidden=128, dropout=0.5):
        super().__init__()
        self.hidden = hidden                                    # <<< store hidden dim
        self.conv = HeteroConv({                                #Processes the heterogeneous knowledge graph with different node types and relationships.
            ('FinancialMetric','QoQ_CHANGE','FinancialMetric'): SAGEConv(in_fm, hidden),
            ('FinancialMetric','BELONGS_TO_PERIOD','TimePeriod'): SAGEConv((in_fm,in_tp), hidden),
            ('TimePeriod','CRITICAL_PERIOD','FinancialMetric'): SAGEConv((in_tp,in_fm), hidden),
            ('TimePeriod','CRITICAL_DAY','StockMetric'):        SAGEConv((in_tp,in_sm), hidden),
            ('TimePeriod','rev_BELONGS','FinancialMetric'):     SAGEConv((in_tp,in_fm), hidden),
        }, aggr='mean')
        self.norm = nn.ModuleDict({
            'FinancialMetric': nn.LayerNorm(hidden),
            'TimePeriod'     : nn.LayerNorm(hidden),
            'StockMetric'    : nn.LayerNorm(hidden),
        })
        self.head = nn.Sequential(
            nn.Linear(hidden*3 + in_gf, hidden//2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden//2,1)
        )

    # Sets up feature dictionaries for each node type and edge types
    def forward(self, data, period_vol):
        T = data['TimePeriod'].num_nodes
        pe = period_emb(torch.arange(T, device=period_vol.device)) * period_vol
        x = {
            'FinancialMetric': data['FinancialMetric'].x,
            'TimePeriod'     : pe,
            'StockMetric'    : data['StockMetric'].x,
        }
        def ei(rel):
            return data[rel].edge_index if rel in data.edge_types else torch.zeros((2,0),dtype=torch.long,device=pe.device)
        edges = {rel: ei(rel) for rel in [
            ('FinancialMetric','QoQ_CHANGE','FinancialMetric'),
            ('FinancialMetric','BELONGS_TO_PERIOD','TimePeriod'),
            ('TimePeriod','CRITICAL_PERIOD','FinancialMetric'),
            ('TimePeriod','CRITICAL_DAY','StockMetric'),
            ('TimePeriod','rev_BELONGS','FinancialMetric'),
        ]}

        # Applies the hetero convolution to the feature dictionaries ←→
        out = self.conv(x, edges)
        fm = F.relu(self.norm['FinancialMetric'](out['FinancialMetric']))
        tp = F.relu(self.norm['TimePeriod'](out['TimePeriod']))
        sm = F.relu(self.norm['StockMetric'](out['StockMetric']))

        fm_pool = fm.mean(dim=0) if fm.size(0)>0 else torch.zeros(self.hidden, device=fm.device)
        tp_pool = tp.mean(dim=0)
        sm_pool = sm.mean(dim=0) if sm.size(0)>0 else torch.zeros(self.hidden, device=sm.device)

        gf      = data['GraphFeature'].x.squeeze(0)
        h       = torch.cat([fm_pool, tp_pool, sm_pool, gf], dim=-1)
        return self.head(h).squeeze()

# ──────────── §7 5-Fold CV for GraphSAGE ─────────────
from torch.optim.lr_scheduler import ReduceLROnPlateau

in_fm = graphs[0]['FinancialMetric'].x.size(1)
in_sm = graphs[0]['StockMetric'].x.size(1)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
sage_acc, sage_auc = [], []

for train_idx, val_idx in kf.split(graphs):
    train_loader = DataLoader([(graphs[i],flags[i]) for i in train_idx], batch_size=1, shuffle=True)
    val_loader   = DataLoader([(graphs[i],flags[i]) for i in val_idx],   batch_size=1)
    model = HeteroSAGENet(in_fm, in_sm, in_tp, in_gf).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = ReduceLROnPlateau(opt,'min',factor=0.5,patience=5)
    pos_w = torch.tensor([Counter(flags[i] for i in train_idx)[0]/
                          Counter(flags[i] for i in train_idx)[1]], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    best_v,pat=1e9,0

    for epoch in range(1,101):
        model.train(); trn=0
        for g,y in train_loader:
            g=g.to(device)
            pv=compute_period_vol(g)
            logits=model(g,pv).unsqueeze(0)
            tgt=torch.tensor([y],dtype=torch.float,device=device)
            loss=loss_fn(logits,tgt)
            opt.zero_grad(); loss.backward(); opt.step()
            trn+=loss.item()
        trn/=len(train_loader)

        model.eval(); vl=0
        with torch.no_grad():
            for g,y in val_loader:
                g=g.to(device)
                pv=compute_period_vol(g)
                loss=loss_fn(model(g,pv).unsqueeze(0),
                             torch.tensor([y],dtype=torch.float,device=device))
                vl+=loss.item()
        vl/=len(val_loader)
        sched.step(vl)
        if vl<best_v:
            best_v,pat,bst=vl,0,model.state_dict()
        else:
            pat+=1
            if pat>=10: break

    model.load_state_dict(bst)
    preds, trues, probs = [],[],[]
    with torch.no_grad():
        for g,y in val_loader:
            g=g.to(device)
            p=torch.sigmoid(model(g,compute_period_vol(g))).item()
            preds.append(int(p>0.5)); trues.append(y); probs.append(p)
    sage_acc.append(accuracy_score(trues,preds))
    sage_auc.append(roc_auc_score(trues,probs))

print("GraphSAGE CV Acc:", np.mean(sage_acc), "±", np.std(sage_acc))
print("GraphSAGE CV AUC:", np.mean(sage_auc), "±", np.std(sage_auc))



from tqdm.auto import tqdm
import logging
from sklearn.model_selection import StratifiedKFold
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import traceback
import matplotlib.pyplot as plt
import seaborn as sns


# ──────────── §8 Define HeteroFTAGNN_v3 + _forward_to_embedding() ─────────────
class HeteroFTAGNN_v3(nn.Module):
    def __init__(self, in_fm,in_sm,in_tp,in_gf,hidden=128,heads=2,dropout=0.2):
        super().__init__()
        self.hidden = hidden

        # 1) define your rel→GATConv dict once:
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
        # save the list of relation‐types for later
        self.rel_keys = list(rel_convs.keys())

        # 2) pass that same mapping into both layers: two layers of GAT first output is used as input for the second layer
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

    def forward(self, data, period_vol):
        h = self._embed(data, period_vol)
        return self.head(h).squeeze(), None, None

    # 1) Prepare Input Features
    def _embed(self, data, period_vol):
        T,dev = data['TimePeriod'].num_nodes, data['FinancialMetric'].x.device
        pe = period_emb(torch.arange(T,device=dev)); pe=torch.nan_to_num(pe)
        x = {
            'FinancialMetric': data['FinancialMetric'].x,
            'TimePeriod'     : pe,
            'StockMetric'    : data['StockMetric'].x,
        }
        #Prepare Edge Indices
        def ei(rel):
            return data[rel].edge_index if rel in data.edge_types else torch.zeros((2,0),dtype=torch.long,device=dev)
        edges = {etype: ei(etype) for etype in self.rel_keys}

        out1 = self.conv1(x, edges) #Apply first GAT layer
        fm1 = F.elu(self.ln['FinancialMetric'](out1['FinancialMetric']))
        tp1 = F.elu(self.ln['TimePeriod'](out1['TimePeriod']))
        sm1 = F.elu(self.ln['StockMetric'](out1['StockMetric']))

        out2 = self.conv2({'FinancialMetric':fm1,'TimePeriod':tp1,'StockMetric':sm1}, edges) #Apply second GAT layer
        fm2 = F.elu(self.ln['FinancialMetric'](out2['FinancialMetric'] + fm1)) #Residual connections: Add Layer 1 output to Layer 2 output
        tp2 = F.elu(self.ln['TimePeriod'](out2['TimePeriod']    + tp1))
        sm2 = F.elu(self.ln['StockMetric'](out2['StockMetric']  + sm1))

        global_vec,_ = self.global_att(tp2,period_vol)
        local_vec,_  = self.local_att(fm2)
        sm_pool      = sm2.mean(dim=0) if sm2.size(0)>0 else torch.zeros(self.hidden,device=dev)
        gf           = data['GraphFeature'].x.squeeze(0)
        return torch.cat([local_vec,global_vec,sm_pool,gf],dim=-1) #Concatenate all embeddings vector

# ──────────── Setup Logging ─────────────
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

# ──────────── §9 Retrain best GNN on full data & extract embeddings ─────────────
model = HeteroFTAGNN_v3(in_fm,in_sm,in_tp,in_gf).to(device)
opt   = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=4.55e-5)
pos_w = torch.tensor([Counter(flags)[0]/Counter(flags)[1]], device=device)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
loader = DataLoader(list(zip(graphs,flags)), batch_size=1, shuffle=True)

# Detect last saved epoch
import os, re

ckpt_dir = "/content/drive/MyDrive/Research/financial_knowledge_graph/epoch"
all_ckpts = os.listdir(ckpt_dir)
epoch_nums = [int(re.search(r"gnn_epoch(\d+)\.pth", f).group(1)) for f in all_ckpts if re.search(r"gnn_epoch(\d+)\.pth", f)]
start_epoch = max(epoch_nums) + 1 if epoch_nums else 1
print(f"Resuming from epoch {start_epoch}")

# Load last checkpoint
if start_epoch > 1:
    last_ckpt_path = f"{ckpt_dir}/gnn_epoch{start_epoch - 1:02d}.pth"
    model.load_state_dict(torch.load(last_ckpt_path))
    print(f"✅ Loaded weights from {last_ckpt_path}")


# Resume training
for epoch in range(start_epoch, 51):
    model.train()
    epoch_loss = 0.0
    for g, y in tqdm(loader, desc=f"Epoch {epoch:02d}", leave=False):
        g = g.to(device)
        pv = compute_period_vol(g)
        logits = model(g, pv)[0].unsqueeze(0)
        tgt    = torch.tensor([y], dtype=torch.float, device=device)
        loss   = loss_fn(logits, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(loader)
    ckpt_path = f"{ckpt_dir}/gnn_epoch{epoch:02d}.pth"
    torch.save(model.state_dict(), ckpt_path)
    logger.info(f"Saved epoch {epoch} checkpoint → {ckpt_path}")
    logger.info(f"Epoch {epoch:02d} completed — avg loss: {avg_loss:.4f}")

# ──────────── Extract GNN embeddings ─────────────
embs, ys = [], []
for g, y in tqdm(zip(graphs, flags),
                 desc="Extracting GNN embeddings",
                 total=len(graphs)):
    model.eval()
    g = g.to(device)
    with torch.no_grad():
        h = model._embed(g, compute_period_vol(g))
    embs.append(h.detach().cpu().numpy())
    ys.append(y)
embs = np.vstack(embs)
print(f"Extracted embeddings for {len(embs)} graphs")

print("✅ Embedding stacking done")

# align wide features matrix
wide = gf_df.set_index('id').loc[ids, feat_cols].values
X, y = np.hstack([wide, embs]), np.array(ys)

print(f"Number of IDs: {len(ids)}")
print(f"Number of rows in gf_df: {len(gf_df)}")
print(f"Missing IDs: {set(ids) - set(gf_df['id'])}")
print(f"Missing feature cols: {set(feat_cols) - set(gf_df.columns)}")

# ──────────── §10 RF Ensemble on [wide | GNN] ─────────────
rf = RandomForestClassifier(n_estimators=200,
                            class_weight='balanced',
                            random_state=42)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
fold_scores = []
print("start randome forest")
for fold, (train_i, val_i) in enumerate(cv.split(X, y), start=1):
    try:
        print(f"RF ensemble — fold {fold}")
        rf.fit(X[train_i], y[train_i])
        preds = rf.predict_proba(X[val_i])[:, 1]
        auc   = roc_auc_score(y[val_i], preds)
        fold_scores.append(auc)
        print(f" Fold {fold} AUC: {auc:.4f}")
    except Exception as e:
        print(f"❌ Error during fold {fold}")
        traceback.print_exc()
        continue

mean_auc = np.mean(fold_scores)
std_auc  = np.std(fold_scores)
print(f"RF+GNN ensemble AUC: {mean_auc:.4f} ± {std_auc:.4f}")

# ──────────── Save final models ─────────────
import joblib
import torch

# 1) Save GNN
gnn_path = '/content/drive/MyDrive/Research/financial_knowledge_graph/best_gnn_v4.pth'
torch.save(model.state_dict(), gnn_path)
print(f"✅ Saved GNN weights to {gnn_path}")

# 2) Save RF + scalers
rf_path     = '/content/drive/MyDrive/Research/financial_knowledge_graph/rf_gnn_ensemble.joblib'
scaler_path = '/content/drive/MyDrive/Research/financial_knowledge_graph/tabular_scaler.joblib'
joblib.dump(rf, rf_path)
print(f"✅ Saved RF ensemble to {rf_path}")

# Save all 3 scalers
tabular_scaler = RobustScaler().fit(gf_df[feat_cols])
joblib.dump(tabular_scaler, '/content/drive/MyDrive/Research/financial_knowledge_graph/scaler/tabular_scaler.joblib')
print("✅ Saved tabular scaler.")

if fm_scaler:
    joblib.dump(fm_scaler, '/content/drive/MyDrive/Research/financial_knowledge_graph/scaler/fm_scaler.joblib')
    print("✅ Saved FinancialMetric scaler.")
if sm_scaler:
    joblib.dump(sm_scaler, '/content/drive/MyDrive/Research/financial_knowledge_graph/scaler/sm_scaler.joblib')
    print("✅ Saved StockMetric scaler.")