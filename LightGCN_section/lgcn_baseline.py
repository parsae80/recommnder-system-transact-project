#!/usr/bin/env python3
import os, math, gc
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from pathlib import Path

# Project root: assumes this file is inside the project repo
PROJECT_ROOT = Path(__file__).resolve().parent

# Main output/data directory inside the repo
DATA_DIR = PROJECT_ROOT / "amazon_personal_care_out"
CLIP_DIR = DATA_DIR / "clip"

INTER_PARQUET = DATA_DIR / "reviews_revised_proc" / "interactions.parquet"

ITEMS_COARSE = DATA_DIR / "items_meta" / "items_features.parquet"

# CLIP dense vectors
CLIP_EMB_PATH = CLIP_DIR / "clip_item_emb.npy"

CLIP_TOPK_PARQUET = CLIP_DIR / "topk_tags.parquet"
# has columns: i, asin, concept, score, rank

ARTIFACTS_DIR = DATA_DIR / "item_tower"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

OUT_EMB_PT = ARTIFACTS_DIR / "item_emb_baseline_no_clip_no_semantic_K3.pt"

SPLIT_DIR = DATA_DIR / "splits"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = SPLIT_DIR / "train.parquet"
VAL_PATH = SPLIT_DIR / "val.parquet"
TEST_PATH = SPLIT_DIR / "test.parquet"
# ------------------ Hparams ------------------
@dataclass
class HParams:
    dim: int = 32
    # Fixed propagation depths trained in series.
    propagation_depths: Tuple[int, ...] = (3,)
    lr: float = 6e-4
    reg: float = 4e-5
    batch_size: int = 256
    batches_per_propagation: int = 64   # >1 speeds up training by reusing one propagation across a small batch chunk
    epochs: int = 100
    seed: int = 42
    run_gc_each_epoch: bool = False            # True only if you hit memory pressure
    lambda_clip: float = 0.0   # baseline: disable CLIP/Semantic contribution
    # add CLIP dense vector
    use_clip_dense: bool = False
    clip_dim: int = 512
    clip_proj_dim: int = 32         # project CLIP -> this dim (should be <= dim)
    clip_dropout: float = 0.0       # dropout on clip proj output

    clip_tau: float = 1.0           # temperature for softmax
    concept_text_emb_path: str = str(CLIP_DIR / "concept_text_emb.npy")
    concept_vocab_path: str = str(CLIP_DIR / "concept_bank.json")
    clip_topk_path: str = str(CLIP_DIR / "topk_tags.parquet")

HP = HParams()

torch.manual_seed(HP.seed)
np.random.seed(HP.seed)

# ---------------- Utils ----------------
def normalize_adj(adj: sp.coo_matrix) -> sp.csr_matrix:
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum + 1e-8, -0.5)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    return (D_inv_sqrt @ adj @ D_inv_sqrt).tocsr()

def build_ui_graph(num_users:int, num_items:int, ui_edges:np.ndarray) -> sp.csr_matrix:
    U, I = num_users, num_items
    rows = np.concatenate([ui_edges[:,0], U + ui_edges[:,1]])
    cols = np.concatenate([U + ui_edges[:,1], ui_edges[:,0]])
    data = np.ones(len(ui_edges)*2, dtype=np.float32)
    A = sp.coo_matrix((data, (rows, cols)), shape=(U+I, U+I))
    return A.tocsr()

def get_stratified_item_indices(train_df):
    """
    Categorizes items into Head, Mid, and Tail based on training popularity.
    
    Returns:
        A dictionary mapping 'head', 'mid', 'tail' to sets of item_ids.
    """
    # 1. Calculate interaction counts per item
    counts = train_df['i'].value_counts().sort_values(ascending=False)
    num_items = len(counts)

    # 2. Determine index cutoffs for 20/30/50 split
    head_cutoff = int(num_items * 0.20)
    mid_cutoff = int(num_items * 0.50)  # Top 20% (Head) + Next 30% (Mid) = Top 50%

    # 3. Create the sets
    stratified_buckets = {
        'head': set(counts.index[:head_cutoff]),
        'mid':  set(counts.index[head_cutoff:mid_cutoff]),
        'tail': set(counts.index[mid_cutoff:])
    }
    
    # 4. Print stats for your thesis logs
    print(f"Stratification Complete:")
    print(f" - Head: {len(stratified_buckets['head'])} items (Avg interactions: {counts.iloc[:head_cutoff].mean():.1f})")
    print(f" - Tail: {len(stratified_buckets['tail'])} items (Avg interactions: {counts.iloc[mid_cutoff:].mean():.1f})")
    
    return stratified_buckets

def time_aware_user_split(
    df: pd.DataFrame,
    user_col: str = "u",
    time_col: str = "time_stamp",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    ensure_min_train: int = 1,
    ensure_val_test_if_possible: bool = True,
):
    """
    Per-user chronological split (no leakage): earliest -> train, then val, then latest -> test.
    """
    required_cols = {user_col, time_col}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for split: {missing}")

    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0, 1)")
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("val_ratio must be in [0, 1)")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1")

    df = df.sort_values([user_col, time_col]).reset_index(drop=True)

    r = df.groupby(user_col).cumcount()
    n = df.groupby(user_col)[user_col].transform("size")

    train_k = (n * train_ratio).astype(np.int64)
    train_k = np.maximum(train_k, ensure_min_train)

    val_k = (n * (train_ratio + val_ratio)).astype(np.int64)
    val_k = np.maximum(val_k, train_k)

    if ensure_val_test_if_possible:
        has3 = n >= 3
        train_k = np.where(has3, np.minimum(train_k, n - 2), train_k)
        val_k = np.where(has3, np.minimum(val_k, n - 1), val_k)

    m_train = r < train_k
    m_val = (r >= train_k) & (r < val_k)
    m_test = r >= val_k

    train_df = df[m_train].copy()
    val_df = df[m_val].copy()
    test_df = df[m_test].copy()

    print(f"Split Complete: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")
    return train_df, val_df, test_df

def _l2_normalize_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)

def _softmax_rows_np(x: np.ndarray, tau: float = 1.0) -> np.ndarray:
    # x: (N, K)
    z = x / max(1e-8, float(tau))
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)

def recall_at_k_sampled(Zu, Zi, val_truth, users, num_items, k=10, num_neg=1000, seed=42):
    rng = np.random.default_rng(seed)
    hits = 0
    for u in users:
        pos_items = list(val_truth[u])
        if len(pos_items) == 0:
            continue

        # pick 1 positive item for evaluation (or loop over all if you want)
        pos = pos_items[0]

        # sample negatives that are NOT in truth
        negs = []
        while len(negs) < num_neg:
            cand = int(rng.integers(0, num_items))
            if cand != pos and cand not in val_truth[u]:
                negs.append(cand)

        candidates = torch.tensor([pos] + negs, device=Zu.device, dtype=torch.long)

        # scores over candidate set only
        scores = (Zu[u] * Zi[candidates]).sum(dim=1)   # (1+num_neg,)
        topk = torch.topk(scores, k).indices
        topk_items = candidates[topk].tolist()

        if pos in topk_items:
            hits += 1

    return hits / max(1, len(users))


def build_clip_boost32_from_topk(
    num_items: int,
    clip_dense_path: str,
    out_dim: int = 32,
    chunk: int = 200_000,
    seed: int = 42,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Returns only the CLIP Visual (Dense) vectors projected to out_dim.
    Also returns the Projection Matrix P to be reused for the semantic vectors.
    """
    # 1. LOAD CLIPS DENSE (Read-only mmap)
    V_dense = np.load(clip_dense_path, mmap_mode="r")
    D = int(V_dense.shape[1])

    # 2. GENERATE SHARED PROJECTION MATRIX
    rng = np.random.default_rng(seed)
    P = rng.standard_normal((D, out_dim), dtype=np.float32)
    # Normalize P columns to keep projection scales stable
    P /= (np.linalg.norm(P, axis=0, keepdims=True) + 1e-8)

    # 3. OUTPUT BUFFER
    out = np.zeros((num_items, out_dim), dtype=np.float32)

    # 4. CHUNK PROCESSING
    for st in range(0, num_items, chunk):
        ed = min(num_items, st + chunk)
        n = ed - st

        # Handle indexing if num_items in inter > V_dense size
        if st >= V_dense.shape[0]:
            vd = np.zeros((n, D), dtype=np.float32)
        else:
            vd = V_dense[st:ed].astype(np.float32, copy=True)
            # Normalize raw 512d CLIP space
            vd /= (np.linalg.norm(vd, axis=1, keepdims=True) + 1e-8)

        # Project to 32d
        z = vd @ P
        # Re-normalize in the 32d subspace
        # New line (fixes broadcasting)
        normalized_z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
        out[st:st + normalized_z.shape[0]] = normalized_z
        
    print(f"[Dense] Processed {num_items} visual vectors. Projection matrix P shape: {P.shape}")
    return torch.from_numpy(out), P


class LightGCN(nn.Module):
    def __init__(self, num_users:int, num_items:int, dim:int, K:int,
                 reg: float, lambda_clip: float=1.0, sem_init: torch.Tensor = None,
                 clip_boost_32: torch.Tensor = None):
        super().__init__()
        self.U, self.I, self.D, self.K = num_users, num_items, dim, K
        self.reg = reg
        self.lambda_clip = float(lambda_clip)
        self.user_emb = nn.Embedding(num_users, dim)
        self.item_id_emb = nn.Embedding(num_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_id_emb.weight, std=0.1)

        # Store CLIP dense vectors as buffer
        if clip_boost_32 is not None:
            self.register_buffer("clip_boost_buffer", clip_boost_32.float())
        else:
            self.clip_boost_buffer = None

        # 2. Store Semantic Buffer
        if sem_init is not None:
            self.register_buffer("clip_sem", sem_init.float())
        else:
            self.clip_sem = None
            print("[Warning] No semantic buffer provided - semantic nudge will be disabled.")
        # # pass semantic buffer from train() if you really want a separate one
        # self.semantic_nudge_buffer = None
        # self.topk_tags_df = topk_tags_df
        # self.seed_concepts = seed_concepts if seed_concepts else set()
        # Store components as separate buffers
        
        
        # 3-way fusion gate: concatenates [v_base, v_dense, v_semantic] -> outputs (alpha, beta, gamma)
        self.fusion_gate = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 2), # output weights for dense and semantic (base is implicit)
            nn.Softmax(dim=1)
        )
        
        # Ablation mode: fixed weights for testing
        self.ablation_weights = None

    @staticmethod
    def compute_semantic_nudge(num_items, topk_path, concept_emb_path, concept_vocab_path, projection_matrix):
        """
        Static method that precomputes semantic nudge buffer (N, 32) where each item is represented by 
        the weighted average of its top-2 CLIP concepts.
        """
        print("[Prep] Vectorizing Semantic Buffer computation...")
        
        # 1. Load data
        v_concepts = torch.from_numpy(np.load(concept_emb_path)).float() # (20017, 512)
        with open(concept_vocab_path, "r") as f:
            data = json.load(f)
        vocab = data["concepts"] if isinstance(data, dict) and "concepts" in data else data
        c2id = {str(c).strip(): i for i, c in enumerate(vocab)}

        # 2. Load and Map Tags
        df = pd.read_parquet(topk_path)
        df = df[df['rank'] <= 2].copy()
        
        # Map concepts to IDs (Vectorized)
        df['concept_id'] = df['concept'].astype(str).str.strip().map(c2id)
        df = df.dropna(subset=['concept_id'])
        
        # 3. Calculate Weights (Vectorized Softmax)
        # We group by 'i' just to get the sum for the denominator
        df['exp_score'] = np.exp(df['score'].values / 0.07)
        denom = df.groupby('i')['exp_score'].transform('sum')
        df['weight'] = df['exp_score'] / (denom + 1e-12)

        # 4. The "Magic" Part: Scattered Addition
        # Instead of looping, we do it all in one pass in PyTorch
        item_indices = torch.from_numpy(df['i'].values).long()
        concept_ids = torch.from_numpy(df['concept_id'].values).long()
        weights = torch.from_numpy(df['weight'].values).float().unsqueeze(1)

        # Get all vectors for all rows in the dataframe at once
        # rows are (10M * rank_limit, 512)
        weighted_vectors = v_concepts[concept_ids] * weights

        # Scatter-add them into the final buffer
        sem_buffer = torch.zeros((num_items, 512))
        sem_buffer.index_add_(0, item_indices, weighted_vectors)
        # Project to 32d using the SAME projection matrix as dense
        if isinstance(projection_matrix, np.ndarray):
            projection_matrix = torch.from_numpy(projection_matrix).float()
            
        sem_buffer_32 = sem_buffer @ projection_matrix.cpu()
        # Normalize (using torch.norm for consistency)
        norm = torch.norm(sem_buffer_32, dim=1, keepdim=True) + 1e-8
        result = (sem_buffer_32 / norm).float()
        print(f"[Prep] Semantic buffer shape: {result.shape}")
        return result


    def fuse_item(self, ids: torch.Tensor) -> torch.Tensor:
        v_base = self.item_id_emb(ids)

        if self.lambda_clip < 1e-6 or self.clip_boost_buffer is None or self.clip_sem is None:
            return v_base
        # clip_boost_buffer should already be (N, dim) and normalized/projection done offline
        v_clip_dense = self.clip_boost_buffer[ids].to(v_base.dtype)
        v_clip_semantic = self.clip_sem[ids].to(v_base.dtype)
        
        gate_input = torch.cat([v_clip_dense, v_clip_semantic], dim=1)
        weights = self.fusion_gate(gate_input)  # (N, 2)
        
        v_content = v_clip_dense * weights[:, 0:1] + v_clip_semantic * weights[:, 1:2]
        # fixed scalar fusion
        return v_base + (self.lambda_clip * v_content)


    def set_ablation_mode(self, alpha: float, beta: float, gamma: float):
        """Set fixed weights for ablation studies. Weights are normalized to sum to 1."""
        total = alpha + beta + gamma
        if total > 0:
            self.ablation_weights = (alpha / total, beta / total, gamma / total)
        else:
            self.ablation_weights = (1/3, 1/3, 1/3)
        print(f"[Ablation] weights = alpha={self.ablation_weights[0]:.4f}, beta={self.ablation_weights[1]:.4f}, gamma={self.ablation_weights[2]:.4f}")

    def disable_ablation_mode(self):
        """Disable ablation mode to learn weights normally."""
        self.ablation_weights = None
        print("[Ablation] disabled - learning weights")

    def propagate(self, A_sp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u0 = self.user_emb.weight
        i0 = self.fuse_item(torch.arange(self.I, device=u0.device))
        X = torch.cat([u0, i0], dim=0)

        # Initialize Z as the first layer (k=0)
        Z = X / (self.K + 1) 
        
        out = X
        for _ in range(self.K):
            out = torch.sparse.mm(A_sp, out)
            # Add to the mean running total to save memory
            Z = Z + (out / (self.K + 1))

        Zu, Zi = Z[:self.U], Z[self.U:]
        return Zu, Zi

    def bpr_loss_cached(self, Zu, Zi, user_ids, pos_i, neg_i):
        u = Zu[user_ids]
        ip = Zi[pos_i]
        ineg = Zi[neg_i]
        x = (u * ip).sum(dim=1) - (u * ineg).sum(dim=1)
        loss = -F.logsigmoid(x).mean()
        loss = loss + self.reg * (u.pow(2).mean() + ip.pow(2).mean() + ineg.pow(2).mean())
        return loss
    @torch.no_grad()
    def export_item_table(self, A_sp: torch.Tensor) -> torch.Tensor:
        _, Zi = self.propagate(A_sp)
        return Zi.detach().cpu()

# --------------- Trainer ---------------
class Sampler:
    def __init__(self, pos_edges: np.ndarray, num_users:int, num_items:int, seed:int=42):
        self.num_users, self.num_items = num_users, num_items
        self.rng = np.random.default_rng(seed)
        self.by_user = [set() for _ in range(num_users)]
        for u, i in pos_edges:
            self.by_user[int(u)].add(int(i))
        self.by_user_arr = [np.fromiter(s, dtype=np.int64) if len(s) > 0 else np.empty(0, dtype=np.int64)
                            for s in self.by_user]
        self.user_pos_counts = np.fromiter(
            (arr.size for arr in self.by_user_arr), dtype=np.int64, count=num_users
        )
        self.user_pos_offsets = np.zeros(num_users + 1, dtype=np.int64)
        np.cumsum(self.user_pos_counts, out=self.user_pos_offsets[1:])
        total_pos = int(self.user_pos_offsets[-1])
        if total_pos > 0:
            self.user_pos_flat = np.concatenate([arr for arr in self.by_user_arr if arr.size > 0])
        else:
            self.user_pos_flat = np.empty(0, dtype=np.int64)
        self.non_empty_users = np.flatnonzero(np.array([len(s) > 0 for s in self.by_user], dtype=np.bool_))

    def _mark_invalid_negatives(self, users: np.ndarray, neg_i: np.ndarray) -> np.ndarray:
        invalid = np.zeros(users.shape[0], dtype=np.bool_)
        unique_users, inverse = np.unique(users, return_inverse=True)
        for group_idx, u in enumerate(unique_users):
            idx = np.flatnonzero(inverse == group_idx)
            pos_items = self.by_user_arr[int(u)]
            if pos_items.size == 0:
                continue
            invalid[idx] = np.isin(neg_i[idx], pos_items, assume_unique=False)
        return invalid

    def batch(self, B:int):
        users = self.rng.integers(0, self.num_users, size=B, endpoint=False)
        # Vectorized positive sampling using flattened user->items index.
        pos_users = users.copy()
        empty_mask = self.user_pos_counts[pos_users] == 0
        if empty_mask.any():
            pos_users[empty_mask] = self.rng.choice(self.non_empty_users, size=int(empty_mask.sum()), replace=True)

        pos_counts = self.user_pos_counts[pos_users]
        pos_offsets = self.user_pos_offsets[pos_users]
        pos_rand = self.rng.random(B)
        pos_idx = (pos_rand * pos_counts).astype(np.int64)
        pos_i = self.user_pos_flat[pos_offsets + pos_idx]

        # Batched negative rejection sampling with grouped membership checks.
        neg_i = self.rng.integers(0, self.num_items, size=B, endpoint=False)
        for _ in range(20):
            invalid = self._mark_invalid_negatives(users, neg_i)
            if not invalid.any():
                break
            neg_i[invalid] = self.rng.integers(0, self.num_items, size=int(invalid.sum()), endpoint=False)

        return (torch.from_numpy(users), torch.from_numpy(pos_i), torch.from_numpy(neg_i))

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- load interactions ----
    inter = pd.read_parquet(INTER_PARQUET)
    NUM_USERS = int(inter["u"].max()) + 1
    NUM_ITEMS = int(inter["i"].max()) + 1
    
    # clip_dense = np.load(CLIP_EMB_PATH , mmap_mode="r")

    if "time_stamp" not in inter.columns:
        raise ValueError("Interactions must contain 'time_stamp' for chronological split")

    pos_inter = inter.copy() # Every row is a positive interaction

    train_df, val_df, test_df = time_aware_user_split(
        pos_inter,
        user_col="u",
        time_col="time_stamp",
        train_ratio=0.8,
        val_ratio=0.1,
        ensure_min_train=1,
        ensure_val_test_if_possible=True,
    )
    
    # After generating train_df, val_df, test_df
    train_df.to_parquet(TRAIN_PATH, index=False)
    val_df.to_parquet(VAL_PATH, index=False)
    test_df.to_parquet(TEST_PATH, index=False)
    print(f"[Storage] Splits saved to {SPLIT_DIR}")

    buckets = get_stratified_item_indices(train_df)
    pos = train_df[["u", "i"]].to_numpy(dtype=np.int64)

    print(
        f"[inter] users={NUM_USERS:,} items={NUM_ITEMS:,} "
        f"train_pos={len(train_df):,} val_pos={len(val_df):,} test_pos={len(test_df):,}"
    )

    # ---- adjacency ----
    A = build_ui_graph(NUM_USERS, NUM_ITEMS, pos)
    A_norm = normalize_adj(A).tocoo()

    indices = torch.from_numpy(np.vstack((A_norm.row, A_norm.col)).astype(np.int64))
    values  = torch.from_numpy(A_norm.data.astype(np.float32))
    A_sp = torch.sparse_coo_tensor(indices, values, (NUM_USERS+NUM_ITEMS, NUM_USERS+NUM_ITEMS)).coalesce()

    # ---- load items + CLIP gating ----
    items = pd.read_parquet(ITEMS_COARSE, columns=["i"])
    if "i" not in items.columns:
        raise ValueError("ITEMS_COARSE must contain column 'i' aligned to item ids")

    clip_boost_32 = None
    topk_tags_df = None
    projection_matrix = None
    sem_buffer = None
    
    if HP.use_clip_dense> 1e-6:
        clip_boost_32, projection_matrix = build_clip_boost32_from_topk(
            num_items=NUM_ITEMS,
            clip_dense_path=CLIP_EMB_PATH,
            out_dim=HP.dim,
            chunk=200_000,
            seed=HP.seed,
        )
        clip_boost_32 = clip_boost_32.to(device, non_blocking=True)
        projection_matrix = torch.from_numpy(projection_matrix).float() if projection_matrix is not None else None
        
        # Load topk_tags for semantic nudge
        if os.path.exists(HP.clip_topk_path):
            topk_tags_df = pd.read_parquet(HP.clip_topk_path)
            print(f"[TopkTags] loaded: shape={topk_tags_df.shape}")
    
        sem_buffer = LightGCN.compute_semantic_nudge(
            num_items=NUM_ITEMS,
            topk_path=HP.clip_topk_path,
            concept_emb_path=HP.concept_text_emb_path,
            concept_vocab_path=HP.concept_vocab_path,
            projection_matrix=projection_matrix
        )
        
    else:
        print("[Mode] Vanilla LightGCN (Baseline). Skipping CLIP/Semantic loading.")
    # ---- model ----
    val_truth = val_df.groupby('u')['i'].apply(set).to_dict()
    val_user_list = list(val_truth.keys())

    k_values = [int(k) for k in HP.propagation_depths]

    if any(k < 0 for k in k_values):
        raise ValueError(f"All propagation depths must be >= 0, got: {k_values}")

    # Preserve order while removing duplicates.
    k_values = list(dict.fromkeys(k_values))
    print(f"[Sweep] Training propagation depths: {k_values}")

    sampler = Sampler(pos, NUM_USERS, NUM_ITEMS, seed=HP.seed)
    A_gpu = A_sp.coalesce().to(device)

    steps_per_epoch = math.ceil(len(pos) / HP.batch_size)

    for k_depth in k_values:
        print(f"\n[Run] Starting training for propagation depth K={k_depth}")

        model = LightGCN(
            num_users=NUM_USERS,
            num_items=NUM_ITEMS,
            dim=HP.dim,
            K=k_depth,
            reg=HP.reg,
            clip_boost_32=clip_boost_32,
            lambda_clip=HP.lambda_clip,
            sem_init=sem_buffer,
        ).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=HP.lr)

        best_recall = 0.0
        PATIENCE = 3
        wait = 0
        best_export_state = None
        last_export_state = None

        for ep in range(1, HP.epochs + 1):
            model.train()
            running = 0.0

            chunk_size = max(1, int(HP.batches_per_propagation))
            for start in range(0, steps_per_epoch, chunk_size):
                cur_chunk = min(chunk_size, steps_per_epoch - start)

                Zu, Zi = model.propagate(A_gpu)
                opt.zero_grad(set_to_none=True)

                chunk_loss = None
                for _ in range(cur_chunk):
                    u, pi, ni = sampler.batch(HP.batch_size)
                    u, pi, ni = u.to(device), pi.to(device), ni.to(device)

                    loss = model.bpr_loss_cached(Zu, Zi, u, pi, ni)
                    chunk_loss = loss if chunk_loss is None else (chunk_loss + loss)
                    running += float(loss.item())

                (chunk_loss / cur_chunk).backward()
                opt.step()

            avg_loss = running / max(1, steps_per_epoch)
            print(f"[K={k_depth}] [epoch {ep}/{HP.epochs}] bpr_loss={avg_loss:.4f}")
            # ---- FULL VALIDATION CHECK (Every 5 Epochs) ----
            if ep % 5 == 0:
                model.eval()
                with torch.no_grad():
                    Zu_val, Zi_val = model.propagate(A_gpu)
                recall_at_10 = recall_at_k_sampled(
                    Zu_val, Zi_val, val_truth, val_user_list, NUM_ITEMS, k=10, num_neg=1000, seed=HP.seed + ep
                )
                last_export_state = {
                    'user_emb': Zu_val.detach().cpu(),
                    'item_emb': Zi_val.detach().cpu(),
                    'epoch': ep,
                    'recall': recall_at_10,
                    'K': int(k_depth),
                }
                print(f" >>> [K={k_depth}] VALIDATION | Sampled Recall@10: {recall_at_10:.4f}")
                
                # ---- SAVE CHECKPOINT AT EACH VALIDATION ----
                stem, ext = os.path.splitext(OUT_EMB_PT)
                validation_ckpt_path = f"{stem}_K{k_depth}_epoch_{ep:03d}{ext}"
                torch.save(last_export_state, validation_ckpt_path)
                print(f"[OK] [K={k_depth}] Saved validation checkpoint to: {validation_ckpt_path}")
                
                # Check for improvement (Overfitting Prevention)
                if recall_at_10 > best_recall:
                    best_recall = recall_at_10
                    wait = 0
                    best_export_state = {
                        'user_emb': Zu_val.detach().cpu(),
                        'item_emb': Zi_val.detach().cpu(),
                        'epoch': ep,
                        'recall': recall_at_10,
                        'K': int(k_depth),
                    }
                else:
                    wait += 1
                    if wait >= PATIENCE:
                        print(f"[K={k_depth}] Early Stopping: Recall hasn't improved for {PATIENCE} validation checks.")
                        break

        # ---- FINAL EXPORT ----
        stem, ext = os.path.splitext(OUT_EMB_PT)
        out_path = f"{stem}_K{k_depth}_final{ext}"
        out_path_best = f"{stem}_K{k_depth}_best{ext}"

        if last_export_state is None:
            raise RuntimeError(f"No validation checkpoint generated for K={k_depth}. Increase epochs or reduce validation interval.")

        if best_export_state is not None:
            torch.save(best_export_state, out_path_best)
            print(f"[OK] [K={k_depth}] Saved best checkpoint to: {out_path_best}")

        torch.save(last_export_state, out_path)
        print(f"[OK] [K={k_depth}] Saved checkpoint to: {out_path}")

        if HP.run_gc_each_epoch:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

if __name__ == "__main__":
    train()
