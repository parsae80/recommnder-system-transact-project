
########################################################################################
# Importing necessary libraries
########################################################################################
# %% [code] cell 2
import tarfile
import os
import pandas as pd, numpy as np, gc
from pathlib import Path
from collections import defaultdict, Counter, deque
import  glob
from dataclasses import dataclass
# Core libs
import logging
import random
# PyTorch
import torch
import torch.nn as nn
# Typing
from typing import Final
from torch.utils.data import Dataset, DataLoader
import math, copy, time

# %% [code] cell 3

# Define paths
# ========= Stream-build ready_seq.train.csv from REVIEWS (memory safe) =========
# Expected columns: reviewerID, asin, overall, unixReviewTime (or reviewTime)
# === paths ===
# Keep all paths in one place so the second-tower setup stays consistent.
# Match the dataset locations used in Untitled-1 (LightGCN training input).from pathlib import Path

# If this file is in the repo root:
PROJECT_ROOT = Path(__file__).resolve().parent

# If this file is inside src/, use this instead:
# PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "amazon_personal_care_out"

INTER_PARQUET = DATA_ROOT / "reviews_revised_proc" / "interactions.parquet"  # from prep_reviews.py
ITEMS_FEAT = DATA_ROOT / "items_meta" / "items_features.parquet"             # from prep_items_meta.py
IDMAP_PARQ = DATA_ROOT / "items_meta" / "item_id_map.parquet"                # (i,parent_asin)

# ---- Load interactions once (Parquet from your prep) ----
inter = pd.read_parquet(INTER_PARQUET).sort_values(["u","time_stamp"]).reset_index(drop=True)
print(inter.head())
# If label missing, derive from rating >= 4

# Required columns
need_cols = {"u","i","time_stamp", "verified_purchase"}
missing = need_cols - set(inter.columns)
if missing:
    raise ValueError(f"INTER_PARQUET missing columns: {missing}")

NUM_USERS = int(inter["u"].max()) + 1
NUM_ITEMS = int(inter["i"].max()) + 1
POS_RATE  = float(inter["verified_purchase"].fillna(0).mean())
print(f"[inter] users={NUM_USERS:,} | items={NUM_ITEMS:,} | rows={len(inter):,} | conv_rate={POS_RATE:.4f}")

############################################################################################
# Forward path wiring
############################################################################################
# Build or load extra features table aligned by 'i'
# LOADING A NEW REVIEW PARQUET WITH THE NEW FEATURES ADDED
AGG_PARQUET = DATA_ROOT / "reviews_revised_proc" / "reviews_agg.parquet"

feat_df = pd.read_parquet(ITEMS_FEAT).sort_values("i").reset_index(drop=True)
print("[debug] items_features columns:", list(feat_df.columns))

if os.path.exists(AGG_PARQUET):
    agg_df = pd.read_parquet(AGG_PARQUET).sort_values("i").reset_index(drop=True)
    # Some pipelines used 'rev_count' vs 'rev_n'; unify to 'rev_count'
    if "rev_n" in agg_df.columns and "rev_count" not in agg_df.columns:
        print("[debug] renaming rev_n -> rev_count in agg_df")
        agg_df = agg_df.rename(columns={"rev_n": "rev_count"})
    feat_df = feat_df.merge(
        agg_df[["i", "rev_verified_ratio", "rev_rating_avg", "rev_count"]],
        on="i", how="left"
    )
else:
    # If missing, create zeros so shape stays consistent (you can error instead if you prefer)
    raise ValueError("File not found:", AGG_PARQUET)
# --- Use exactly the extra cols you intend to feed ---
EXTRA_COLS = [
    "img_count", "has_images", "avg_rating_meta", "rating_count_meta", 
    "price_num", "pop_log",
    "rev_count", "rev_verified_ratio", "rev_rating_avg" # The NEW ones
]
missing = [c for c in EXTRA_COLS if c not in feat_df.columns]
if missing:
    raise ValueError(f"Missing columns for wide features: {missing}")

device = "cuda" if torch.cuda.is_available() else "cpu"
extra_feat_in_dim = len(EXTRA_COLS)

########################################################################################
# Hyperparameters Configurations
########################################################################################

# %% [code] cell 6
USER_COL = "user"
ITEM_COL = "adgroup_id"
TIME_COL = "time_stamp"   # UNIX seconds
LABEL_COL = "clk"         # 0/1

# Sequence building
SEQ_LEN = 100
MIN_CLICK_HISTORY = 1      # require at least this many prior clicks to form a history


########################################################################################
# #  Build Amazon sequential dataset + DataLoaders
########################################################################################

# Training subset controls (optional)
BATCH_SIZE = 32


########################################################################################
# Transact config
########################################################################################

# %% [code] cell 12

@dataclass
class TransActConfig:
    """
    Configuration class to build a TransAct PyTorch module.

    :param seq_len: Length of the input sequence
    :param time_window_ms: Time window in milliseconds for random window mask
    :param latest_n_emb: Number of latest embeddings to use in output
    :param concat_candidate_emb: Whether to concatenate candidate embeddings with user sequence
    :param concat_max_pool: Whether to apply max pooling to the output of the transformer encoder and append it to output
    :param action_vocab: Vocabulary of user actions
    :param action_emb_dim: Dimension of user action embeddings
    :param item_emb_dim: Dimension of item embeddings
    :param num_layer: Number of TransformerEncoderLayer
    :param nhead: Number of heads in the TransformerEncoderLayer
    :param dim_feedforward: Feed forward dimension of the TransformerEncoderLayer
    """

    seq_len: int = 100
    time_window_ms: int = 1000 * 60 * 60 * 1
    latest_n_emb: int = 10
    concat_candidate_emb: bool = True
    concat_max_pool: bool = True
    action_vocab: list = range(0, 20)
    action_emb_dim: int = 32
    item_emb_dim: int = 32
    num_layer: int = 2
    nhead: int = 2
    dim_feedforward: int = 32
    #added for the wide layer
    extra_feat_in_dim: int = 9  # number of extra features to expect as input
    extra_feat_dim: int = 16 # number of extra features to concatenate +5 to make it dividible 
    concat_extra_feat_to_tf: bool = True     # add it to the Transformer input (per step)
    concat_extra_feat_to_head: bool = True   # also concatenate at the very end (to CTR head)

########################################################################################
# Build the transformer(transact) network
########################################################################################

# %% [code] cell 14
class TransAct(nn.Module):
    """
    TransAct: Transformer-based Realtime User Action Model for Recommendation at Pinterest
    (Backbone; scoring head is external.)
    """
    
    def __init__(self, transact_config: TransActConfig):
        super().__init__()
        self.transact_config = transact_config
        self.action_emb_dim = transact_config.action_emb_dim
        self.item_emb_dim   = transact_config.item_emb_dim
        self.seq_len        = transact_config.seq_len
        self.latest_n_emb   = transact_config.latest_n_emb
        self.time_window_ms = transact_config.time_window_ms
        self.concat_candidate_emb = transact_config.concat_candidate_emb
        self.action_vocab:   Final[list] = self.transact_config.action_vocab  # e.g., [0, 1, 2] for no rate>=4 / rate>=4 unverified/ verified>=4
        
        # Single projector used for both deep (TF) and wide (head) paths
        self.extra_feat_proj = nn.Linear(
            transact_config.extra_feat_in_dim,
            transact_config.extra_feat_dim
        )

        # ---- compute Transformer input size (d_model) once
        base_dim = self.action_emb_dim + self.item_emb_dim       # action + history item
        if self.concat_candidate_emb:
            base_dim += self.item_emb_dim                        # + candidate item
        if transact_config.concat_extra_feat_to_tf:
            base_dim += transact_config.extra_feat_dim           # + projected features

        self.transformer_in_dim = base_dim

        assert (self.transformer_in_dim % transact_config.nhead) == 0, \
        f"d_model={self.transformer_in_dim} must be divisible by nhead={transact_config.nhead}"

        # embeddings + encoder
        self.register_buffer("action_type_lookup", self.convert_vocab_to_idx())
        self.action_emb_module = nn.Embedding(
            num_embeddings=len(list(transact_config.action_vocab)) + 1,
            embedding_dim=self.action_emb_dim,
            padding_idx=0,
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.transformer_in_dim,
            nhead=transact_config.nhead,
            dim_feedforward=transact_config.dim_feedforward,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=transact_config.num_layer)

        if transact_config.concat_max_pool:
            self.out_linear = nn.Linear(self.transformer_in_dim, self.transformer_in_dim)
        else:
            self.out_linear = None


    def convert_vocab_to_idx(self) -> torch.Tensor:
        """
        Map raw action ids to embedding indices with PAD=0.
        Assumes input action_type_seq uses -1 for PAD, so index with (a + 1):
        -1+1 -> 0 (PAD),  0+1 -> 1 (first real action), etc.
        """
        logging.info(f"Action used: {self.action_vocab}")
        t = torch.zeros(100, dtype=torch.long)  # original size; enlarge if needed
        i = 1  # start from 1 so PAD stays 0
        for aid in sorted(self.action_vocab):
            idx = aid + 1
            if idx >= t.numel():
                t = torch.cat([t, torch.zeros(idx - t.numel() + 1, dtype=torch.long)], dim=0)
            t[idx] = i
            i += 1
        return t

    def _adjust_mask(self, pad_mask: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
        """
        Combine padding and time-window masks for src_key_padding_mask.
        Returns BoolTensor where True => position is ignored by attention.
        """
        mask = torch.bitwise_or(pad_mask, time_mask)
        if mask.size(1) > 0:
            mask[:, 0] = False  # ensure at least one token visible
        return mask  # BoolTensor

    def forward(
        self,
        action_type_seq: torch.Tensor,    # (B,S)  -1=PAD, 0=impr, 1=click (2=purchase if you add CVR)
        item_embedding_seq: torch.Tensor, # (B,S,D)
        action_time_seq: torch.Tensor,    # (B,S)  IMPORTANT: same units as request_time
        request_time: torch.Tensor,       # (B,1) or (B,) or (B,1,1)
        item_embedding: torch.Tensor,
        extra_feat: torch.Tensor = None,  # (B,D)
    ) -> torch.Tensor:

        # step 1: clip to seq_len
        action_type_seq    = action_type_seq[:, : self.seq_len]
        item_embedding_seq = item_embedding_seq[:, : self.seq_len, :]
        action_time_seq    = action_time_seq[:, : self.seq_len]

        # step 2: action embedding (PAD->0; real actions -> 1..K)
        at_idx = self.action_type_lookup[action_type_seq + 1]
        action_emb_tensor = self.action_emb_module(at_idx)  # (B,S,A)

        # step 3: key padding mask (Bool), plus time-window mask
        pad_mask = (at_idx == 0)  # True where PAD (impressions are NOT PAD)

        # normalize request_time to (B,S)
        req = request_time
        if req.dim() == 3 and req.size(-1) == 1:  # (B,1,1)->(B,1)
            req = req.squeeze(-1)
        if req.dim() == 1:                        # (B,)->(B,1)
            req = req.unsqueeze(1)
        req = req.expand(-1, self.seq_len)        # (B,1)->(B,S)

        # IMPORTANT: ensure req and action_time_seq have the SAME units (ms or sec)
        rand_time_window_ms = random.randint(0, self.time_window_ms)
        short_time_window_idx_trn  = (req - action_time_seq) < rand_time_window_ms
        short_time_window_idx_eval = (req - action_time_seq) < 0

        if self.training:
            key_padding_mask = self._adjust_mask(pad_mask, short_time_window_idx_trn)
        else:
            key_padding_mask = self._adjust_mask(pad_mask, short_time_window_idx_eval)
        # key_padding_mask is BoolTensor (True => ignore)

        # step 4: stack embeddings
        # added part for wide layer
        inputs = [action_emb_tensor, item_embedding_seq]  # (B,S, A + D)

        if self.transact_config.concat_candidate_emb:
            cand_expanded = item_embedding.unsqueeze(1).expand(-1, self.seq_len, -1)
            inputs.append(cand_expanded)

        # 🔧 PROJECT ONCE and REUSE
        projected_feat = None
        if extra_feat is not None:
            projected_feat = self.extra_feat_proj(extra_feat)  # (B, extra_feat_dim)

        # Inject into Transformer input if enabled
        if self.transact_config.concat_extra_feat_to_tf and projected_feat is not None:
            assert projected_feat is not None, "extra_feat required when concat_extra_feat_to_tf=True"
            extra_expanded = projected_feat.unsqueeze(1).expand(-1, self.seq_len, -1)
            inputs.append(extra_expanded)

        if self.transact_config.concat_extra_feat_to_tf:
            assert extra_feat is not None, "concat_extra_feat_to_tf=True but extra_feat is None"


        action_pin_emb = torch.cat(inputs, dim=-1)  # (B,S, d_model)
        # step 5: transformer
        tfmr_out = self.transformer_encoder(
            src=action_pin_emb,
            src_key_padding_mask=key_padding_mask  # Bool mask
        )

        # step 6: output packing
        output_concat = []
        
        # A) Max Pooling signal
        if self.transact_config.concat_max_pool:
            pooled_out = self.out_linear(tfmr_out.max(dim=1).values)
            output_concat.append(pooled_out)
        
        # B) Flattened Sequence signal (Latest N)
        if self.latest_n_emb > 0:
            seq_out = tfmr_out[:, : self.latest_n_emb].flatten(1)
        else:
            seq_out = tfmr_out.flatten(1)
        output_concat.append(seq_out)

        # C) WIDE CONNECTION: Shortcut the 11 features directly to the output
        if self.transact_config.concat_extra_feat_to_head:
            assert extra_feat is not None, "extra_feat must be provided when concat_extra_feat_to_head=True"
            extra_proj = self.extra_feat_proj(extra_feat)          # (B, extra_feat_dim)
            output_concat.append(extra_proj)

        return torch.cat(output_concat, dim=1)
        
    @property
    def output_dim(self) -> int:
        # number of features in the final output vector (32 +32 + 32 + 11 = 107)
        H = self.transformer_in_dim
        # number of sequence embeddings used
        N = self.transact_config.latest_n_emb if self.transact_config.latest_n_emb > 0 else self.transact_config.seq_len
        total_dim = (H if self.out_linear is not None else 0) + (N*H)
        # added part for wide layer
        if self.transact_config.concat_extra_feat_to_head:
            total_dim += self.transact_config.extra_feat_dim
            
        return total_dim

########################################################################################
# Instantiating the config
########################################################################################

# %% [code] cell 16
# Example: adjust to your dataset
# - SEQ_LEN: whatever you used when creating sequences (e.g., 20)
# - ITEM_EMB_DIM: the dimensionality of your item embeddings/table (e.g., 64)

cfg = TransActConfig(
    seq_len=SEQ_LEN,                # <- from your dataset pipeline
    item_emb_dim= 32,      # <- your item embedding width
    action_vocab=[0, 1, 2],            
    time_window_ms=120_000,          # 1 minute; match your timestamps' unit
    latest_n_emb=5,
    concat_candidate_emb=True,
    num_layer=2,
    nhead=4,
    dim_feedforward=128,
    extra_feat_in_dim=extra_feat_in_dim,
    extra_feat_dim=16,
    concat_extra_feat_to_head=True,
    concat_extra_feat_to_tf=True)

encoder = TransAct(cfg).to(device)
print("TransAct output_dim:", encoder.output_dim)

print(f"[sanity] d_model={encoder.transformer_in_dim} "
    f"extra_in={cfg.extra_feat_in_dim} extra_proj={cfg.extra_feat_dim} "
    f"concat_tf={cfg.concat_extra_feat_to_tf} concat_head={cfg.concat_extra_feat_to_head} "
    f"output_dim={encoder.output_dim}")

# ========= Item embeddings: train-or-load (feature-aware LightGCN) =========
ITEM_EMB_DIM = 32  # must match cfg.item_emb_dim
# Option A: load pretrained item embeddings (shape [NUM_ITEMS, ITEM_EMB_DIM])
ITEM_TOWER_DIR = DATA_ROOT / "item_tower"

LGCN_EMB_PT_CLIP = ITEM_TOWER_DIR / "item_emb_with_clip_dense_plus_semantic_checkpoints.pt"
LGCN_EMB_PT_BASELINE = ITEM_TOWER_DIR / "item_emb_baseline_no_clip_no_semantic.pt"

# Pick which item embedding to use for the second tower.
ITEM_EMB_PT = LGCN_EMB_PT_CLIP

if not ITEM_EMB_PT:
    raise FileNotFoundError("ITEM_EMB_PT is empty; set it to a valid checkpoint path.")

if not os.path.exists(ITEM_EMB_PT):
    raise FileNotFoundError(f"Item embedding checkpoint not found: {ITEM_EMB_PT}")

if ITEM_EMB_PT and os.path.exists(ITEM_EMB_PT):
    _obj = torch.load(ITEM_EMB_PT, map_location="cpu")
    if isinstance(_obj, dict) and "item_emb" in _obj:
        _w = _obj["item_emb"].float()
    else:
        _w = _obj.float()
    assert _w.shape == (NUM_ITEMS, ITEM_EMB_DIM), f"Bad item emb shape: {_w.shape}"
    item_model = nn.Embedding(NUM_ITEMS, ITEM_EMB_DIM, _weight=_w).to(device)
    item_model.weight.requires_grad_(False)  # freeze stage-1
    print("[info] Loaded pretrained item embeddings from", ITEM_EMB_PT)
print("Item model ready:", item_model)

def set_requires_grad(module, value: bool):
    for param in module.parameters():
        param.requires_grad = value

########################################################################################
# The added part for scoring head
########################################################################################

# %% [code] cell 20
# ========= Scoring head + item table + optimizer =========
# ========= Scoring head + optimizer + loss + tuple builder + loaders =========

# (1) Scoring head and optimizer
#added part for wide layer
ctr_head = nn.Linear(encoder.output_dim, 1).to(device)

print("[sanity]", "d_model=", encoder.transformer_in_dim,
    "output_dim=", encoder.output_dim)

# Stage 1: freeze Transformer + CLIP item encoder; train wide + head
set_requires_grad(encoder, False)
if hasattr(encoder, "extra_feat_proj"):
    set_requires_grad(encoder.extra_feat_proj, True)
set_requires_grad(item_model, False)

def build_optimizer():
    params = [p for p in encoder.parameters() if p.requires_grad]
    params += [p for p in item_model.parameters() if p.requires_grad]
    params += [p for p in ctr_head.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=1e-4)

# IMPORTANT: Re-initialize optimizer AFTER unfreezing
opt = build_optimizer()

def build_lr_scheduler(optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        threshold=1e-4,
        min_lr=1e-6,
    )

# Reduce LR when validation loss plateaus
lr_scheduler = build_lr_scheduler(opt)

# Calculate weight: num_negatives / num_positives
# Based on your log's 75.6% pos_rate: 0.2439 / 0.7561 approx 0.322
pos_ratio = 0.756 
weight_val = (1 - pos_ratio) / pos_ratio
pos_weight = torch.tensor([weight_val]).to(device)

# Replace your standard BCE with this:
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# bce = nn.BCEWithLogitsLoss(reduction="mean")

grad_debug_done = False

def _grad_norm(params):
    norms = [p.grad.norm().item() for p in params if p.grad is not None]
    return float(np.mean(norms)) if norms else float("nan")

# (2) Utilities for batch forward
@torch.no_grad()
def _make_masked_seq_item_emb(seq_items_t):
    seq_items = seq_items_t.to(device)
    mask = (seq_items >= 0).unsqueeze(-1)             # (B,S,1)
    seq_items_clamped = seq_items.clamp_min(0)        # -1 -> 0
    seq_item_emb = item_model(seq_items_clamped) * mask
    return seq_item_emb

def forward_batch(batch):
    (seq_items_t, seq_times_t, act_types_t, req_time_t, cand_i_t, label_t) = batch
    B = seq_items_t.size(0)

    seq_item_emb = _make_masked_seq_item_emb(seq_items_t)      # (B,S,D)
    cand_ids = cand_i_t.to(device).view(B)
    
    # One-time runtime check
    if cfg.concat_extra_feat_to_head and not hasattr(encoder, "_wide_checked"):
        B = cand_ids.shape[0]
        extra_feat = extra_table[cand_ids]  # (B, F_in)
        print(f"[wide/debug] first-batch extra_feat shape={tuple(extra_feat.shape)} "
            f"min={float(extra_feat.min()):.3f} max={float(extra_feat.max()):.3f} "
            f"any_nan={torch.isnan(extra_feat).any().item()}")
        # Keep a flag so we only print once
        encoder._wide_checked = True

    cand_emb = item_model(cand_ids)                             # (B,D)
    #gather wide features
    extra_feat = extra_table[cand_ids]          # (B,11)
    extra_feat = extra_feat.to(device)
    
    if extra_feat is not None and extra_feat.dim() == 2:
        assert extra_feat.size(1) == cfg.extra_feat_in_dim, \
        f"extra_feat columns {extra_feat.size(1)} != cfg.extra_feat_in_dim {cfg.extra_feat_in_dim}"

    seq_times_ms = seq_times_t.to(device)                       # already ms if you built it so
    req_time_ms  = req_time_t.to(device).long()                 # (B,1)
    act_types    = act_types_t.to(device)

    enc_out = encoder(
        action_type_seq=act_types,
        item_embedding_seq=seq_item_emb,
        action_time_seq=seq_times_ms,
        request_time=req_time_ms,
        item_embedding=cand_emb,
        extra_feat=extra_feat,
        )
    logits = ctr_head(enc_out).squeeze(-1)
    labels = label_t.to(device).squeeze(-1).float()
    return logits, labels

# (3) Build tuples from interactions (conversion: verified_purchase == 1)
def build_review_tuples(frame: pd.DataFrame,
                        seq_len: int = 100,
                        k_rand_neg: int = 0,
                        k_pop_neg: int = 0,
                        popular_pool: np.ndarray = None,
                        seed: int = 42):
    """
    Emits tuples:
    (seq_items, seq_times_ms, seq_action_types, req_time_ms, cand_i, y)

    Target y remains your CTR-like label: 1 if verified_purchase else 0 for the *current* item.
    
    Adds hard negatives for positive targets:
    - Popular negatives: sample from 'popular_pool' (global top-K) not in user's history
    """
    rng = np.random.default_rng(seed)
    tuples = []
    N_items = int(frame["i"].max()) + 1 if len(frame) else 0

    if "verified_purchase" not in frame.columns:
        raise ValueError("frame must have 'verified_purchase' for conversion labeling")
    
            
    # Safe defaults
    popular_pool = popular_pool if popular_pool is not None else np.arange(N_items, dtype=np.int64)
    for _, g in frame.groupby("u", sort=False):
        g = g.sort_values("time_stamp")

        # ----- build arrays/lists once per user -----
        items   = g["i"].astype(np.int64).to_numpy()
        times_s = g["time_stamp"].astype(np.int64).to_numpy()
        verified = g["verified_purchase"].fillna(0).astype(np.int8).to_numpy()

        # conversion label and action ids (0 = no conversion, 1 = conversion)
        labels_bin = verified.astype(np.int64).tolist()
        act_ids = labels_bin

        items_list   = items.tolist()
        times_list   = times_s.tolist()     # seconds
        # act_ids already list
        # labels_bin already list

        # ----- emit tuples -----
        for t in range(1, len(items_list)):
            hist_i = items_list[:t]
            hist_t = times_list[:t]
            hist_a = act_ids[:t]           # <-- list; safe to slice

            if not hist_i:
                continue

            pad = max(0, seq_len - len(hist_i))
            seq_i = ([-1]*pad) + hist_i[-seq_len:]
            seq_t = ([0]*pad)  + [int(x)*1000 for x in hist_t[-seq_len:]]  # ms
            seq_a = ([-1]*pad) + hist_a[-seq_len:]

            req_ms = int(times_list[t]) * 1000
            y      = int(labels_bin[t])
            cand_i = int(items_list[t])

            # ground-truth tuple
            tuples.append((seq_i, seq_t, seq_a, req_ms, cand_i, y))

            # negatives only when positive target
            if y != 1 or N_items <= 0:
                continue

            excl = set(hist_i[-seq_len:])
            excl.add(cand_i)

            # (A) popular hard negatives
            if k_pop_neg > 0 and popular_pool.size > 0:
                tries = 0; added = 0
                while added < k_pop_neg and tries < k_pop_neg * 30:
                    tries += 1
                    j = int(popular_pool[rng.integers(0, len(popular_pool))])
                    if j not in excl:
                        tuples.append((seq_i, seq_t, seq_a, req_ms, j, 0))
                        excl.add(j); added += 1

            # (B) random negatives
            if k_rand_neg > 0:
                tries = 0; added = 0
                while added < k_rand_neg and tries < k_rand_neg * 30:
                    tries += 1
                    j = int(rng.integers(0, N_items))
                    if j not in excl:
                        tuples.append((seq_i, seq_t, seq_a, req_ms, j, 0))
                        excl.add(j); added += 1

    return tuples

# (4) Train/Val/Test split (match Untitled-1)
SPLIT_DIR = DATA_ROOT / "splits"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = SPLIT_DIR / "train.parquet"
VAL_PATH = SPLIT_DIR / "val.parquet"
TEST_PATH = SPLIT_DIR / "test.parquet"

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

if os.path.exists(TRAIN_PATH) and os.path.exists(VAL_PATH) and os.path.exists(TEST_PATH):
    train_df = pd.read_parquet(TRAIN_PATH)
    val_df = pd.read_parquet(VAL_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    print(f"[Split] Loaded splits from {SPLIT_DIR}")
else:
    if "time_stamp" not in inter.columns:
        raise ValueError("Interactions must contain 'time_stamp' for chronological split")
    pos_inter = inter.copy()
    train_df, val_df, test_df = time_aware_user_split(
        pos_inter,
        user_col="u",
        time_col="time_stamp",
        train_ratio=0.8,
        val_ratio=0.1,
        ensure_min_train=1,
        ensure_val_test_if_possible=True,
    )
    train_df.to_parquet(TRAIN_PATH, index=False)
    val_df.to_parquet(VAL_PATH, index=False)
    test_df.to_parquet(TEST_PATH, index=False)
    print(f"[Storage] Splits saved to {SPLIT_DIR}")

# ---- Build wide feature table using train-only stats (avoid leakage) ----
train_item_ids = set(train_df["i"].unique())
if not train_item_ids:
    raise ValueError("train_df has no items; cannot compute wide feature stats")

extra_df = feat_df[EXTRA_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32")
train_mask = feat_df["i"].isin(train_item_ids)
train_extra_df = extra_df[train_mask]

wide_mu, wide_sigma = {}, {}
for c in EXTRA_COLS:
    s = train_extra_df[c]
    mu, sigma = float(s.mean()), float(s.std() + 1e-8)
    extra_df[c] = (extra_df[c] - mu) / sigma
    wide_mu[c] = mu
    wide_sigma[c] = sigma

extra_table = torch.tensor(extra_df.values, dtype=torch.float32).to(device)
extra_table = torch.nan_to_num(extra_table, nan=0.0, posinf=0.0, neginf=0.0)
assert torch.isfinite(extra_table).all(), "[wide] extra_table contains non-finite values"

print(f"[wide/debug] items_features has columns: {list(feat_df.columns)[:8]}... (+{len(feat_df.columns)-8} more)")
print(f"[wide/debug] USING WIDE_COLS ({len(EXTRA_COLS)}): {EXTRA_COLS}")
print(f"[wide/debug] extra_table shape={tuple(extra_table.shape)}  any_nan={np.isnan(extra_df.values).any()}")
print(f"[wide] using EXTRA_COLS={EXTRA_COLS}")
print(f"[wide] extra_table.shape={tuple(extra_table.shape)}  extra_feat_in_dim={extra_feat_in_dim}")
# (5) tuples (no extra negatives for AUC/PR) ====
#########################################
# added part for the negative pool
##########################################
# ---- Prepare negative pools (popular + category) ----
# /home/p_esla/rec_proj/hyper_tune/transact_hard_negatives_hypert.py
def prepare_negative_pools(inter_df: pd.DataFrame,
                        topk_pop: int = 5000):
    """
    Returns:
    popular_pool: np.array of top-K popular item ids (by positive count)
    item2cat: dict[int -> int or str] item -> category id/label
    cat2items: dict[cat -> np.array of item ids]
    """
    # 1) Popular items by positive count
    if "verified_purchase" in inter_df.columns:
        pop_series = inter_df[inter_df["verified_purchase"] == 1].groupby("i").size().sort_values(ascending=False)
    else:
        # fallback if conversion missing: treat all interactions as positives
        pop_series = inter_df.groupby("i").size().sort_values(ascending=False)
    popular_pool = pop_series.head(topk_pop).index.to_numpy(dtype=np.int64)

    return popular_pool

POPULAR_TOPK = 2000
POPULAR_POOL_PATH = os.path.join(
    DATA_ROOT,
    f"popular_pool_top{POPULAR_TOPK}.npy",
)
if os.path.exists(POPULAR_POOL_PATH):
    popular_pool = np.load(POPULAR_POOL_PATH)
    print(f"Loaded popular_pool from {POPULAR_POOL_PATH} ({len(popular_pool):,} items)")
else:
    popular_pool = prepare_negative_pools(
        inter, topk_pop=POPULAR_TOPK
    )
    np.save(POPULAR_POOL_PATH, popular_pool)
    print(f"Prepared popular_pool and saved to {POPULAR_POOL_PATH} ({len(popular_pool):,} items)")

SEQ_LEN = int(getattr(cfg, "seq_len", 100))

TRAIN_TUPLES = build_review_tuples(
    train_df,
    seq_len=SEQ_LEN,
    k_rand_neg=3,
    k_pop_neg=0,      # try 1–5
    popular_pool=popular_pool,
)
# TRAIN_TUPLES = build_review_tuples(train_df, seq_len=SEQ_LEN, k_rand_neg=0)
VALID_TUPLES = build_review_tuples(val_df,   seq_len=SEQ_LEN, k_rand_neg=0)
TEST_TUPLES = build_review_tuples(test_df, seq_len=SEQ_LEN, k_rand_neg=0)
print(f"TRAIN_TUPLES={len(TRAIN_TUPLES):,} | VALID_TUPLES={len(VALID_TUPLES):,} | TEST_TUPLES={len(TEST_TUPLES):,}")


# (6) Datasets & DataLoaders
class ReviewSeqDataset(Dataset):
    def __init__(self, tuples): self.tuples = tuples
    def __len__(self): return len(self.tuples)
    def __getitem__(self, idx):
        seq_items, seq_times, act_types, req_ms, cand_i, y = self.tuples[idx]
        return (
            torch.tensor(seq_items, dtype=torch.long),
            torch.tensor(seq_times, dtype=torch.long),
            torch.tensor(act_types, dtype=torch.long),
            torch.tensor([req_ms], dtype=torch.long),
            torch.tensor([cand_i], dtype=torch.long),
            torch.tensor([y], dtype=torch.float32),
        )

train_loader = DataLoader(
    ReviewSeqDataset(TRAIN_TUPLES),
    batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
    pin_memory=True, num_workers=2, persistent_workers=True
)
valid_loader = DataLoader(
    ReviewSeqDataset(VALID_TUPLES),
    batch_size=BATCH_SIZE, shuffle=False,
    pin_memory=True, num_workers=2, persistent_workers=True
)
test_loader = DataLoader(
    ReviewSeqDataset(TEST_TUPLES),
    batch_size=BATCH_SIZE, shuffle=False,
    pin_memory=True, num_workers=2, persistent_workers=True
)
print(f"test_loader={len(test_loader)}")

print(f"train_loader={len(train_loader)} | valid_loader={len(valid_loader)}")


########################################################################################
# Training loop with validation metrics
########################################################################################
# 1) One epoch of train or eval (same as yours, minus HIT@3)
def run_epoch(loader, train=True):
    global grad_debug_done
    if train:
        encoder.train(); item_model.train(); ctr_head.train()
    else:
        encoder.eval();  item_model.eval();  ctr_head.eval()

    total_loss, total_count = 0.0, 0
    pos_sc_sum = pos_sc_n = 0.0
    neg_sc_sum = neg_sc_n = 0.0

    for batch in loader:
        logits, labels = forward_batch(batch)
        loss = criterion(logits, labels)

        if train:
            opt.zero_grad()
            loss.backward()
            if not grad_debug_done:
                enc_g = _grad_norm(encoder.parameters())
                item_g = _grad_norm(item_model.parameters())
                head_g = _grad_norm(ctr_head.parameters())
                print(f"[grad/debug] encoder={enc_g:.4e} item_model={item_g:.4e} ctr_head={head_g:.4e}")
                grad_debug_done = True
            opt.step()

        bs = labels.numel()
        total_loss  += loss.item() * bs
        total_count += bs

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            pos_mask = (labels == 1)
            neg_mask = (labels == 0)
            if pos_mask.any():
                pos_sc_sum += probs[pos_mask].sum().item()
                pos_sc_n   += pos_mask.sum().item()
            if neg_mask.any():
                neg_sc_sum += probs[neg_mask].sum().item()
                neg_sc_n   += neg_mask.sum().item()

    avg_loss = total_loss / max(1, total_count)
    pos_mean = (pos_sc_sum / max(1, pos_sc_n)) if pos_sc_n else 0.0
    neg_mean = (neg_sc_sum / max(1, neg_sc_n)) if neg_sc_n else 0.0
    return avg_loss, pos_mean, neg_mean

# 2) Validation metrics: ROC-AUC, PR-AUC, log-loss

@torch.no_grad()
def evaluate_auc_logloss(loader, max_batches=None):
    from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

    encoder.eval(); item_model.eval(); ctr_head.eval()

    y_true, y_prob = [], []
    total_loss, total_count = 0.0, 0
    bce = nn.BCEWithLogitsLoss(reduction="mean")

    # --- STEP 2: CALIBRATION PARAMS ---
    # train_pos_ratio: the 75.6% from your logs
    # real_world_ratio: a realistic estimate (e.g., 1%)
    p_train = 0.756
    p_real  = 0.01
    logit_shift = np.log(p_real / (1 - p_real)) - np.log(p_train / (1 - p_train))
    #
    for bi, batch in enumerate(loader):
        logits, labels = forward_batch(batch)

        # debug: check finiteness before sigmoid
        bad_logit_mask = ~torch.isfinite(logits)
        if bad_logit_mask.any():
            idx = bad_logit_mask.nonzero(as_tuple=False).squeeze(-1)
            print(f"[DEBUG] batch {bi}: non-finite logits at idx={idx.tolist()}")
            print("  sample logits:", logits[idx][:5])
            # sanitize
            logits[bad_logit_mask] = 0.0

        calibrated_logits = logits + logit_shift
        probs = torch.sigmoid(calibrated_logits) # already clamped earlier



        # extra debug: check probs
        bad_prob_mask = ~torch.isfinite(probs)
        if bad_prob_mask.any():
            idx = bad_prob_mask.nonzero(as_tuple=False).squeeze(-1)
            print(f"[DEBUG] batch {bi}: non-finite probs at idx={idx.tolist()}")
            print("  sample probs:", probs[idx][:5])
            probs[bad_prob_mask] = 0.5

        # accumulate
        y_true.append(labels.detach().cpu().numpy().ravel())
        y_prob.append(probs.detach().cpu().numpy().ravel())

        # loss for reference
        total_loss  += bce(logits, labels).item() * labels.numel()
        total_count += labels.numel()

        if max_batches is not None and (bi + 1) >= max_batches:
            break

    if not y_true:
        return float("nan"), float("nan"), float("nan")

    y_true = np.concatenate(y_true) 
    y_prob = np.concatenate(y_prob)

    # final guard before sklearn
    if not np.isfinite(y_prob).all():
        bad = np.where(~np.isfinite(y_prob))[0]
        print(f"[DEBUG] y_prob contains non-finite values at idx[0:20]={bad[:20].tolist()}")
        # replace with 0.5 to proceed
        y_prob = np.nan_to_num(y_prob, nan=0.5, posinf=1.0, neginf=0.0)

    eps = 1e-7
    _ = log_loss(y_true, np.clip(y_prob, eps, 1 - eps), labels=[0, 1])

    if y_true.min() == y_true.max():
        auc = float("nan"); ap = float("nan")
    else:
        auc = roc_auc_score(y_true, y_prob)
        ap  = average_precision_score(y_true, y_prob)

    val_loss = total_loss / max(1, total_count)
    return val_loss, auc, ap


# 3) Train a few epochs and report CTR metrics
EPOCHS = 100          # start small; scale up once it’s stable
VAL_EVERY = 5        # run validation + save checkpoint every N epochs
MAX_EVAL_BATCHES = 100   # or set e.g. 200 if your valid set is huge to save RAM/time

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "new_amazon_dataset"
    / "transact_phase"
    / "clip_transact"
    / "clip_enriched_checkpoints"
)

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
RUN_ID = time.strftime("%Y%m%d_%H%M%S")


ES_METRIC   = "val_loss"   # choose: "val_loss" | "roc_auc" | "pr_auc"
PATIENCE    = 5            # stop if no improvement for this many epochs
MIN_DELTA   = 1e-4         # minimum change to count as improvement

# Initialize tracking
best_val_loss = math.inf
best_auc      = -math.inf
best_pr       = -math.inf
wait = 0
best_ckpt = None
unfreeze_done = False

for ep in range(EPOCHS):
    tr_loss, tr_pos, tr_neg = run_epoch(train_loader, train=True)

    if (not unfreeze_done) and (ep + 1) >= 5:
        print("[unfreeze] epoch>=5; unfreezing Transformer then CLIP encoder")
        set_requires_grad(encoder, True)
        set_requires_grad(item_model, True)
        opt = build_optimizer()
        lr_scheduler = build_lr_scheduler(opt)
        unfreeze_done = True

    if (ep + 1) % VAL_EVERY == 0:
        va_loss, va_auc, va_ap = evaluate_auc_logloss(valid_loader, max_batches=MAX_EVAL_BATCHES)

        lr_scheduler.step(va_loss)

        print(
            f"Epoch {ep+1}/{EPOCHS} | "
            f"train loss={tr_loss:.4f}  pos_prob~={tr_pos:.3f}  neg_prob~={tr_neg:.3f} || "
            f"valid loss={va_loss:.4f}  ROC-AUC={va_auc:.4f}  PR-AUC={va_ap:.4f}"
        )

        ckpt_path = os.path.join(
            CHECKPOINT_DIR,
            f"ckpt_clip_enriched_ep{ep+1:03d}_{RUN_ID}.pt",
        )
        torch.save(
            {
                "encoder":    {k: v.cpu() for k, v in encoder.state_dict().items()},
                "item_model": {k: v.cpu() for k, v in item_model.state_dict().items()},
                "ctr_head":   {k: v.cpu() for k, v in ctr_head.state_dict().items()},
                "metrics": {
                    "epoch": ep + 1,
                    "train_loss": float(tr_loss),
                    "val_loss": float(va_loss),
                    "roc_auc": float(va_auc),
                    "pr_auc": float(va_ap),
                },
                "cfg": {
                    "item_emb_dim": int(item_model.embedding_dim),
                    "seq_len": int(SEQ_LEN),
                    "time_window_ms": int(cfg.time_window_ms),
                    "latest_n_emb": int(cfg.latest_n_emb),
                    "concat_candidate_emb": bool(cfg.concat_candidate_emb),
                    "concat_max_pool": bool(cfg.concat_max_pool),
                    "action_vocab": list(cfg.action_vocab),
                    "action_emb_dim": int(cfg.action_emb_dim),
                    "num_layer": int(cfg.num_layer),
                    "nhead": int(cfg.nhead),
                    "dim_feedforward": int(cfg.dim_feedforward),
                    "concat_extra_feat_to_tf": bool(cfg.concat_extra_feat_to_tf),
                    "concat_extra_feat_to_head": bool(cfg.concat_extra_feat_to_head),
                    "extra_feat_in_dim": int(cfg.extra_feat_in_dim),
                    "extra_feat_dim": int(cfg.extra_feat_dim),
                },
            },
            ckpt_path,
        )
        print(f"Saved validation checkpoint to {ckpt_path}")

        # Decide improvement based on the chosen metric
        if ES_METRIC == "val_loss":
            improved = (va_loss < best_val_loss - MIN_DELTA)
        elif ES_METRIC == "roc_auc":
            improved = (not math.isnan(va_auc)) and (va_auc > best_auc + MIN_DELTA)
        else:  # "pr_auc"
            improved = (not math.isnan(va_ap)) and (va_ap > best_pr + MIN_DELTA)

        if improved:
            # reset patience & stash best values/weights
            wait = 0
            if ES_METRIC == "val_loss":
                best_val_loss = va_loss
            elif ES_METRIC == "roc_auc":
                best_auc = va_auc
            else:
                best_pr = va_ap

            best_ckpt = {
                "encoder":    copy.deepcopy(encoder.state_dict()),
                "item_model": copy.deepcopy(item_model.state_dict()),
                "ctr_head":   copy.deepcopy(ctr_head.state_dict()),
                # (optional) "opt": copy.deepcopy(opt.state_dict()),
            }
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"Early stopping at epoch {ep+1} (no improvement for {PATIENCE} validations).")
                break
    else:
        print(
            f"Epoch {ep+1}/{EPOCHS} | "
            f"train loss={tr_loss:.4f}  pos_prob~={tr_pos:.3f}  neg_prob~={tr_neg:.3f}"
        )

# Restore best weights (optional but recommended)
if best_ckpt is not None:
    encoder.load_state_dict(best_ckpt["encoder"])
    item_model.load_state_dict(best_ckpt["item_model"])
    ctr_head.load_state_dict(best_ckpt["ctr_head"])
    print("Restored best checkpoint.")

    wide_meta = {
        "wide_cols": list(EXTRA_COLS),
        "wide_mu":   wide_mu,
        "wide_sigma": wide_sigma,
    }
    final_path = os.path.join(
        os.path.dirname(CHECKPOINT_DIR),
        "clip_transact_final_best.pt",
    )
    torch.save({
        "encoder":    {k: v.cpu() for k, v in best_ckpt["encoder"].items()},
        "item_model": {k: v.cpu() for k, v in best_ckpt["item_model"].items()},
        "ctr_head":   {k: v.cpu() for k, v in best_ckpt["ctr_head"].items()},
        # (optional) include shapes & hparams you’ll need to rebuild modules
        "cfg": {
            "item_emb_dim": int(item_model.embedding_dim),
            "seq_len": int(SEQ_LEN),
            "time_window_ms": int(cfg.time_window_ms),
            "latest_n_emb": int(cfg.latest_n_emb),
            "concat_candidate_emb": bool(cfg.concat_candidate_emb),
            "concat_max_pool": bool(cfg.concat_max_pool),
            "action_vocab": list(cfg.action_vocab),
            "action_emb_dim": int(cfg.action_emb_dim),
            "num_layer": int(cfg.num_layer),
            "nhead": int(cfg.nhead),
            "dim_feedforward": int(cfg.dim_feedforward),
            "concat_extra_feat_to_tf": bool(cfg.concat_extra_feat_to_tf),
            "concat_extra_feat_to_head": bool(cfg.concat_extra_feat_to_head),
            "extra_feat_in_dim": int(cfg.extra_feat_in_dim),
            "extra_feat_dim": int(cfg.extra_feat_dim),
        },
        "wide_meta": wide_meta,
    }, final_path)
    print(f"Saved final best checkpoint to {final_path}")
