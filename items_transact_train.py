"""
Single-file items pipeline:
- Trains feature-aware LightGCN on (u,i,clk) + item side features.
- Exports/loads item embedding matrix W.
- Provides a TransAct-style embedding module with a single trainable DEFAULT row for OOV.
- Exposes ensure_item_model(...) to train-or-load and return a ready-to-use item model.

Usage in main_transact.py
-------------------------
from items import ensure_item_model
item_model = ensure_item_model(
    interactions_csv="data/ready_subset.csv",
    items_csv="data/items_meta.csv",
    artifacts_dir="artifacts",
    dim=64, K=2, epochs=5,
    cat_cols=["category_id","brand_id"],
    num_cols=["price"],
    device="cuda",
)

# Then in your forward:
# cand_ids_raw: (B,), seq_items_raw: (B,S) with PAD=-1
cand_idx = item_model.map_raw_ids(cand_ids_raw)
cand_emb = item_model.embed_single(cand_idx)
seq_idx  = item_model.map_raw_ids_for_seq(seq_items_raw)
seq_emb  = item_model.embed_seq(seq_idx)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Literal
import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ------------------------------
# Feature prep helpers
# ------------------------------
def _prep_item_features(items_df: pd.DataFrame, cat_cols: List[str], num_cols: List[str]):
    """Encode categoricals (with UNK=0) and standardize numerics.
    Returns (cat_maps, cat_cards, num_norm, items_df_with_encoded_cols).
    """
    cat_maps: Dict[str, Dict[int,int]] = {}
    cat_cards: Dict[str, int] = {}
    items_df = items_df.copy()

    for c in cat_cols:
        vals = items_df[c].astype("Int64")
        uniq_sorted = sorted(set(int(x) for x in vals.dropna().unique()))
        mp = {v: i+1 for i, v in enumerate(uniq_sorted)}  # 0 reserved for UNK/missing
        cat_maps[c] = mp
        cat_cards[c] = len(mp) + 1
        items_df[c] = vals.map(lambda x: mp.get(int(x), 0) if pd.notna(x) else 0).astype("int64")

    num_norm = {}
    for c in num_cols:
        x = items_df[c].astype("float32")
        mu = float(np.nanmean(x)) if len(x) else 0.0
        sd = float(np.nanstd(x)) if len(x) else 1.0
        sd = max(sd, 1e-6)
        items_df[c] = ((x.fillna(mu) - mu) / sd).astype("float32")
        num_norm[c] = {"mean": mu, "std": sd}

    return cat_maps, cat_cards, num_norm, items_df

# ------------------------------
# Feature-aware LightGCN (item-only vectors)
# ------------------------------
class FeatureLightGCN(nn.Module):
    def __init__(self, num_users: int, num_items: int, dim: int, K: int, adj,
                 cat_cards: Dict[str,int], num_dims: int, cat_cols: List[str]):
        super().__init__()
        self.num_users, self.num_items, self.K = num_users, num_items, K
        self.adj = adj
        self.cat_cols = cat_cols
        self.dim = dim

        # ID embeddings (users+items)
        self.id_emb = nn.Embedding(num_users + num_items, dim)
        nn.init.normal_(self.id_emb.weight, std=0.01)

        # Item feature encoders
        self.cat_embs = nn.ModuleDict()
        cat_dim_each = max(8, dim // 8)
        feat_in = 0
        for c, card in cat_cards.items():
            emb = nn.Embedding(card, cat_dim_each)
            nn.init.normal_(emb.weight, std=0.02)
            self.cat_embs[c] = emb
            feat_in += cat_dim_each
        feat_in += num_dims

        self.feat_proj = nn.Sequential(nn.Linear(feat_in, dim), nn.ReLU(inplace=True))
        self.mix_gate = nn.Parameter(torch.zeros(dim))  # sigmoid → [0,1]

    def _item_feat_vecs(self, items_cat: Dict[str, torch.Tensor], items_num: torch.Tensor):
        parts = [self.cat_embs[c](items_cat[c]) for c in self.cat_cols]
        if items_num is not None:
            parts.append(items_num)
        x = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
        return self.feat_proj(x)

    def propagate(self, x):
        outs, h = [x], x
        for _ in range(self.K):
            h = torch.sparse.mm(self.adj, h)
            outs.append(h)
        return torch.stack(outs, dim=0).mean(dim=0)

    def forward(self, items_cat: Dict[str, torch.Tensor], items_num: torch.Tensor):
        x0 = self.id_emb.weight  # (U+I, D)
        U = self.num_users
        feat_i = self._item_feat_vecs(items_cat, items_num)  # (I,D)
        id_i = x0[U:]
        gate = torch.sigmoid(self.mix_gate).unsqueeze(0)
        x0 = x0.clone()
        x0[U:] = gate * id_i + (1 - gate) * feat_i
        return self.propagate(x0)

    def split(self, z):
        return z[: self.num_users], z[self.num_users :]

# ------------------------------
# TransAct-style item module (with DEFAULT row)
# ------------------------------
@dataclass
class ItemEmbConfig:
    num_items: int
    emb_dim: int
    pad_id: int = -1
    l2_normalize: bool = False
    use_layernorm: bool = False
    dropout_p: float = 0.0
    device: str = "cpu"

class ItemModelWithDefault(nn.Module):
    def __init__(self, cfg: ItemEmbConfig, pretrained_W: np.ndarray | None,
                 init_default: Literal["mean","zeros"] = "mean"):
        super().__init__()
        self.cfg = cfg
        self.num_items = int(cfg.num_items)
        self.emb_dim = int(cfg.emb_dim)
        self.default_index = self.num_items
        self.table = nn.Embedding(self.num_items + 1, self.emb_dim)
        with torch.no_grad():
            if pretrained_W is None:
                nn.init.normal_(self.table.weight, std=0.02)
            else:
                W = torch.as_tensor(pretrained_W, dtype=torch.float32)
                assert W.shape == (self.num_items, self.emb_dim)
                self.table.weight[:self.num_items].copy_(W)
                if init_default == "mean":
                    self.table.weight[self.default_index].copy_(W.mean(dim=0))
                else:
                    self.table.weight[self.default_index].zero_()
        self.ln = nn.LayerNorm(self.emb_dim) if cfg.use_layernorm else nn.Identity()
        self.drop = nn.Dropout(cfg.dropout_p) if cfg.dropout_p > 0 else nn.Identity()
        self.to(cfg.device)
        self.item2idx: Dict[int,int] = {}

    def _post(self, x):
        x = self.ln(x); x = self.drop(x)
        if self.cfg.l2_normalize:
            x = x / (x.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        return x

    def map_raw_ids(self, raw_ids: torch.Tensor) -> torch.Tensor:
        arr = raw_ids.detach().cpu().numpy()
        mapper = np.frompyfunc(lambda x: self.item2idx.get(int(x), self.default_index), 1, 1)
        mapped = mapper(arr).astype(np.int64)
        return torch.from_numpy(mapped).to(raw_ids.device)

    def map_raw_ids_for_seq(self, raw_seq_ids: torch.Tensor) -> torch.Tensor:
        pad = self.cfg.pad_id
        x = raw_seq_ids.clone()
        mask = (x != pad)
        if mask.any():
            x[mask] = self.map_raw_ids(x[mask])
        return x

    def embed_single(self, item_ids_idx: torch.Tensor) -> torch.Tensor:
        x = self.table(item_ids_idx.view(-1))
        return self._post(x)

    def embed_seq(self, seq_ids_idx: torch.Tensor) -> torch.Tensor:
        mask = (seq_ids_idx >= 0).unsqueeze(-1)
        x = self.table(seq_ids_idx.clamp_min(0)) * mask
        return self._post(x)

# ------------------------------
# Train-or-load orchestration (single call from main)
# ------------------------------

def ensure_item_model(
    interactions_csv: str,
    items_csv: str,
    artifacts_dir: str = "artifacts",
    *,
    dim: int = 64,
    K: int = 2,
    epochs: int = 5,
    lr: float = 1e-3,
    neg_per_pos: int = 5,
    batch_users: int = 4096,
    cat_cols: List[str] | None = None,
    num_cols: List[str] | None = None,
    pad_id: int = -1,
    init_default: Literal["mean","zeros"] = "mean",
    l2_normalize: bool = False,
    use_layernorm: bool = False,
    dropout_p: float = 0.0,
    device: str = "cuda",
) -> ItemModelWithDefault:
    """Train (if missing) and return a ready-to-use item module with default OOV row.
    Artifacts saved to {artifacts_dir}/items.npy and items_meta.json
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    npy_path  = os.path.join(artifacts_dir, "items.npy")
    meta_path = os.path.join(artifacts_dir, "items_meta.json")

    if not (os.path.exists(npy_path) and os.path.exists(meta_path)):
        # ---- Train feature-aware LightGCN ----
        device_t = torch.device(device if torch.cuda.is_available() else "cpu")
        cats = cat_cols or []
        nums = num_cols or []

        # Interactions: keep clk==1
        df = pd.read_csv(interactions_csv, usecols=["u","i","clk"]) \
               .query("clk == 1")[ ["u","i"] ].drop_duplicates().reset_index(drop=True)
        raw_u = df["u"].to_numpy(np.int64)
        raw_i = df["i"].to_numpy(np.int64)
        uniq_u, u_idx = np.unique(raw_u, return_inverse=True)
        uniq_i, i_idx = np.unique(raw_i, return_inverse=True)
        num_users, num_items = len(uniq_u), len(uniq_i)
        item2idx = {int(k): int(v) for v, k in enumerate(uniq_i)}

        # Join features, aligned to uniq_i order
        items = pd.read_csv(items_csv)
        assert "i" in items.columns, "items_csv must include 'i'"
        items = items.set_index("i").reindex(uniq_i).reset_index()
        cat_maps, cat_cards, num_norm, items_enc = _prep_item_features(items, cats, nums)

        # Build normalized adjacency for bipartite graph
        u = torch.from_numpy(u_idx)
        i = torch.from_numpy(i_idx) + num_users
        N = num_users + num_items
        src = torch.cat([u, i]); dst = torch.cat([i, u])
        ones = torch.ones_like(src, dtype=torch.float32)
        A = torch.sparse_coo_tensor(torch.stack([src, dst]), ones, (N, N))
        deg = torch.sparse.sum(A, dim=1).to_dense() + 1e-8
        d_inv_sqrt = torch.pow(deg, -0.5)
        vals = d_inv_sqrt[src] * ones * d_inv_sqrt[dst]
        DAD = torch.sparse_coo_tensor(torch.stack([src, dst]), vals, (N, N)).coalesce().to(device_t)

        # Feature tensors for items 0..I-1
        items_cat = {c: torch.tensor(items_enc[c].to_numpy(np.int64), device=device_t) for c in cats}
        items_num = torch.tensor(items_enc[nums].to_numpy(np.float32), device=device_t) if nums else None

        model = FeatureLightGCN(num_users, num_items, dim, K, DAD, {k:int(v) for k,v in cat_cards.items()}, len(nums), cats).to(device_t)
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        # Build positives per user
        user_pos = [[] for _ in range(num_users)]
        for ui, ii in zip(u_idx, i_idx):
            user_pos[int(ui)].append(int(ii))
        user_pos_sets = [set(lst) for lst in user_pos]
        rng = np.random.default_rng(123)
        num_steps = max(1, num_users // batch_users)

        def sample_triples(batch_users_np, neg_k):
            users, pos_items, neg_items = [], [], []
            for u_ in batch_users_np:
                if not user_pos[u_]:
                    continue
                pi = rng.choice(user_pos[u_])
                pos_set = user_pos_sets[u_]
                for _ in range(neg_k):
                    while True:
                        nj = int(rng.integers(0, num_items))
                        if nj not in pos_set:
                            break
                    users.append(u_); pos_items.append(pi); neg_items.append(nj)
            if not users:
                return None
            return (
                torch.tensor(users, dtype=torch.long, device=device_t),
                torch.tensor(pos_items, dtype=torch.long, device=device_t),
                torch.tensor(neg_items, dtype=torch.long, device=device_t),
            )

        for ep in range(epochs):
            model.train(); total = 0.0
            for _ in range(num_steps):
                batch_u = rng.integers(0, num_users, size=batch_users)
                sample = sample_triples(batch_u, neg_per_pos)
                if sample is None: continue
                u_t, pi_t, ni_t = sample
                z = model(items_cat, items_num)   # (U+I, D)
                Ue, Ie = model.split(z)
                pos = (Ue[u_t] * Ie[pi_t]).sum(-1)
                neg = (Ue[u_t] * Ie[ni_t]).sum(-1)
                bpr = -torch.log(torch.sigmoid(pos - neg) + 1e-8).mean()
                opt.zero_grad(); bpr.backward(); opt.step()
                total += bpr.item()
            print(f"[Items] Epoch {ep+1}/{epochs} BPR={total/max(1,num_steps):.4f}")

        with torch.no_grad():
            z = model(items_cat, items_num)
            _, Ie = model.split(z)
            W = Ie.detach().cpu().numpy().astype("float32")

        # Save artifacts
        np.save(npy_path, W)
        with open(meta_path, "w") as f:
            json.dump({
                "num_items": int(W.shape[0]),
                "item2idx": {int(k):int(v) for k,v in item2idx.items()},
                "cat_cols": cats,
                "num_cols": nums,
                "num_norm": num_norm,
            }, f)
    else:
        W = np.load(npy_path)
        with open(meta_path) as f:
            meta = json.load(f)
        item2idx = {int(k): int(v) for k, v in meta["item2idx"].items()}

    # Build item module with DEFAULT row
    if 'meta' not in locals():
        with open(meta_path) as f:
            meta = json.load(f)
    cfg = ItemEmbConfig(num_items=int(W.shape[0]), emb_dim=int(W.shape[1]), pad_id=pad_id,
                        l2_normalize=l2_normalize, use_layernorm=use_layernorm,
                        dropout_p=dropout_p, device=device)
    item_model = ItemModelWithDefault(cfg, pretrained_W=W, init_default=init_default)
    item_model.item2idx = {int(k): int(v) for k, v in meta["item2idx"].items()}
    return item_model
