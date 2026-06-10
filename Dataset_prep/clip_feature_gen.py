#!/usr/bin/env python3
# clip_open_vocab_enrichment_ann.py
# Scalable open-vocab CLIP enrichment with FAISS ANN kNN (no full NxN).

import os, re, json, pathlib, math, collections
from typing import List, Tuple, Iterable
import numpy as np
import pandas as pd

# ========= PATHS & KNOBS (edit as needed) =========
from pathlib import Path

# If this file is inside src/, use parents[1].
# If this file is in the repo root, use .parent.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "amazon_personal_care_out"

ITEMS_PARQUET = DATA_ROOT / "items_meta" / "items_features.parquet"  # must have: asin, title

OUT_DIR = DATA_ROOT / "clip"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MEDIA_DIR = DATA_ROOT / "downloaded_media_all"

MANIFEST_JSONL = MEDIA_DIR / "images_per_asin.jsonl"
IMAGE_ROOT = MEDIA_DIR

MAKE_CLIP      = True           # compute CLIP embeddings
TOPK_TAGS      = 10             # top-K concepts per item
TOPK_NEIGHBORS = 10             # top-K image ANN neighbors
BATCH_IMG      = 128            # image encoding batch size
BATCH_TXT      = 256            # text encoding batch size
BATCH_SCORE    = 4096           # scoring batch for image->concept sims (rows per step)

# ANN index knobs (tune for your scale/hardware)
FAISS_USE_GPU  = True           # try faiss-gpu if available
IVF_NLIST      = 4096           # number of inverted lists (coarse centroids)
PQ_M           = 64             # product-quantization sub-vectors (D must be divisible by M)
IVF_TRAIN_MAX  = 500_000        # max samples to train IVF/PQ
IVF_PROBE      = 16             # nprobe at query time
# ==================================================

# ---- optional CLIP ----
try:
    import torch, open_clip
    from PIL import Image
    HAS_CLIP = True
except Exception:
    HAS_CLIP = False

# ---- optional FAISS ----
try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

def _ensure_dir(p): os.makedirs(p, exist_ok=True); return p
def _exists_nonempty(p: str) -> bool: return os.path.exists(p) and os.path.getsize(p) > 0

def norm_text(x: str) -> str:
    x = (x or "").strip()
    x = re.sub(r"\s+", " ", x)
    return x

def _read_manifest(jsonl_path: str) -> pd.DataFrame:
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line: 
                continue  # Skip empty lines
            
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping corrupt JSON on line {line_num}")
                continue

            # 1. Look for ID
            asin = j.get("asin") or j.get("parent_asin") or j.get("item_id")
            
            # 2. Look for path - using 'or []' prevents the NoneType error
            path_source = j.get("paths") or j.get("image_path") or j.get("path") or []
            
            # 3. Force it into a list format
            if isinstance(path_source, str):
                path_source = [path_source]

            chosen = ""
            # 4. Safe to loop now because path_source is guaranteed to be a list
            for p in path_source:
                if _exists_nonempty(str(p)):
                    chosen = str(p)
                    break
            
            if asin and chosen:
                rows.append({"asin": str(asin), "image_path": chosen})
            
    return pd.DataFrame(rows)

def _find_local_image(asin: str) -> str:
    shard = os.path.join(IMAGE_ROOT, asin[:2], asin)
    if os.path.isdir(shard):
        for p in sorted(os.listdir(shard)):
            fp = os.path.join(shard, p)
            if _exists_nonempty(fp): return fp
    try:
        for p in pathlib.Path(IMAGE_ROOT).rglob(f"{asin}*"):
            if p.is_file() and p.stat().st_size > 0:
                return str(p)
    except Exception:
        pass
    return ""

def load_items_with_images() -> pd.DataFrame:
    df = pd.read_parquet(ITEMS_PARQUET)
    if 'parent_asin' in df.columns and 'asin' not in df.columns:
        df = df.rename(columns={'parent_asin': 'asin'})
    for col in ["asin","title"]:
        if col not in df.columns:
            raise ValueError(f"{ITEMS_PARQUET} must contain column '{col}'")
    items = df[["asin","title"]].copy()
    items["asin"]  = items["asin"].astype(str)
    items["title"] = items["title"].astype(str)
    man = _read_manifest(MANIFEST_JSONL)
    merged = items.merge(man, on="asin", how="left")

    need = merged["image_path"].isna() | (merged["image_path"].astype(str)=="")
    if need.any():
        fills = []
        for a in merged.loc[need, "asin"].tolist():
            fills.append(_find_local_image(a))
        merged.loc[need, "image_path"] = fills

    merged["image_path"] = merged["image_path"].astype(str)
    merged = merged[merged["image_path"].apply(_exists_nonempty)].reset_index(drop=True)
    merged.insert(0, "i", np.arange(len(merged), dtype=np.int32))
    return merged[["i","asin","title","image_path"]]

# ---------- concept mining (open vocabulary) ----------
STOPWORDS = set("""
a an the of and or for with on to from by in into over under at as is are be been being it this that those these
your you we they he she them his her their our its
""".split())
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+\-_.']+")

def tokenize(txt: str):
    txt = norm_text(txt).lower()
    return [t for t in TOKEN_RE.findall(txt) if t not in STOPWORDS and len(t) >= 2]

def mine_concepts(titles: Iterable[str], max_vocab=20_000, min_df=5) -> List[str]:
    unigrams = collections.Counter()
    bigrams  = collections.Counter()
    for t in titles:
        toks = tokenize(t)
        unigrams.update(set(toks))
        bigrams.update(set(zip(toks, toks[1:])))
    uni = [w for w, c in unigrams.items() if c >= min_df]
    bi  = [" ".join(bg) for bg, c in bigrams.items() if c >= min_df]
    vocab = uni + bi
    vocab = sorted(vocab, key=lambda s: (-len(s.split()), s))
    return vocab[:max_vocab]

SEED_CONCEPTS = [
    "pump bottle", "squeeze tube", "spray mist", "dropper bottle", "jar with lid",
    "matte finish", "frosted glass", "transparent plastic", "metallic packaging",
    "thick cream", "clear liquid", "foaming gel", "colored liquid",
    "minimalist aesthetic", "vibrant colorful design", "clinical medical look", "botanical floral patterns"
]
QUALITY_HEADS = [
    "a high quality studio product photo",
    "a blurry low quality product photo",
]

# ---------- CLIP ----------
def build_clip(device="cuda"):
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    dev = device if ("cuda" in device and torch.cuda.is_available()) else "cpu"
    return model.to(dev).eval(), preprocess, dev

def encode_texts(texts: List[str], tokenizer, model, device, batch_size=BATCH_TXT) -> np.ndarray:
    out = []
    with torch.no_grad():
        for s in range(0, len(texts), batch_size):
            chunk = texts[s:s+batch_size]
            if not chunk: break
            toks = tokenizer(chunk).to(device)
            emb = model.encode_text(toks)
            emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
            out.append(emb.float().cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, model.text_projection.shape[1]), dtype="float32")

def open_image(path):
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception:
        return None

def encode_images(paths: List[str], preprocess, model, device, batch_size=BATCH_IMG, mmap_path=None, dim_hint=512):
    """
    Returns a memory-mapped array (or ndarray) of shape [N, D] with L2-normalized embeddings.
    """
    N = len(paths)
    if N == 0:
        return np.zeros((0, dim_hint), dtype=np.float32)
    # prepare output mmap to avoid peak memory
    if mmap_path:
        arr = np.memmap(mmap_path, mode="w+", dtype="float32", shape=(N, dim_hint))
    else:
        arr = np.zeros((N, dim_hint), dtype=np.float32)

    cursor = 0
    with torch.no_grad():
        for s in range(0, N, batch_size):
            batch_paths = paths[s:s+batch_size]
            imgs = []
            keep = []
            for idx, p in enumerate(batch_paths):
                im = open_image(p)
                if im is None: continue
                imgs.append(preprocess(im)); keep.append(idx)
            if keep:
                tensor = torch.stack(imgs, dim=0).to(device)
                emb = model.encode_image(tensor)
                emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
                emb = emb.float().cpu().numpy()
            else:
                emb = np.zeros((0, dim_hint), dtype="float32")

            # place into output; fill missing with zeros
            out_chunk = np.zeros((len(batch_paths), emb.shape[-1] if emb.size else dim_hint), dtype="float32")
            for k, idx in enumerate(keep):
                out_chunk[idx] = emb[k]
            arr[s:s+len(batch_paths)] = out_chunk
            cursor += len(batch_paths)
    return arr

# ---------- utilities ----------
def cosine_topk_rows_batched(A: np.ndarray, B: np.ndarray, topk: int, batch_rows: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute top-k of A @ B^T batched over rows of A only (never materialize full sim).
    A: [N, D], B: [M, D]
    Returns (vals[N, k], idx[N, k])
    """
    N = A.shape[0]
    k = min(topk, B.shape[0])
    vals_out = np.empty((N, k), dtype=np.float32)
    idx_out  = np.empty((N, k), dtype=np.int32)
    for s in range(0, N, batch_rows):
        e = min(s + batch_rows, N)
        sims = A[s:e] @ B.T  # [b, M]
        # partial topk per row
        part_idx = np.argpartition(-sims, kth=k-1, axis=1)[:, :k]
        part_val = np.take_along_axis(sims, part_idx, axis=1)
        # sort inside
        order = np.argsort(-part_val, axis=1)
        vals  = np.take_along_axis(part_val, order, axis=1)
        idxs  = np.take_along_axis(part_idx, order, axis=1)
        vals_out[s:e] = vals
        idx_out[s:e]  = idxs.astype(np.int32)
    return vals_out, idx_out

# ---------- FAISS ANN ----------
def build_faiss_index(x: np.ndarray, use_gpu=True, nlist=4096, m=64, train_max=500_000, nprobe=16):
    """
    x: L2-normalized float32 embeddings [N, D]
    Returns (index, is_gpu)
    """
    if not HAS_FAISS:
        raise RuntimeError("faiss is not installed. Install faiss-cpu or faiss-gpu.")
    N, D = x.shape
    quantizer = faiss.IndexFlatIP(D)  # cosine via dot (since normalized)
    index = faiss.IndexIVFPQ(quantizer, D, nlist, m, 8)  # 8 bits per sub-quantizer (tune if needed)

    # train on a random subset
    rng = np.random.default_rng(123)
    train_idx = rng.choice(N, size=min(train_max, N), replace=False)
    index.train(x[train_idx])

    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            is_gpu = True
        except Exception:
            is_gpu = False
    else:
        is_gpu = False

    index.nprobe = nprobe
    index.add(x)  # add all (on GPU if moved)
    return index, is_gpu

def ann_self_search(index, x: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search each vector against the index; remove self when present.
    Returns (scores[N, k], ids[N, k]) using IP (cosine with normalized x).
    """
    k_req = topk + 1  # ask one extra to drop self
    scores, ids = index.search(x, k_req)  # [N, k+1]
    # drop self id if present in row
    out_ids = np.empty((x.shape[0], topk), dtype=np.int32)
    out_sc  = np.empty((x.shape[0], topk), dtype=np.float32)
    for r in range(x.shape[0]):
        row_ids = ids[r]
        row_sc  = scores[r]
        # filter out self (index r)
        keep = [(iid, sc) for iid, sc in zip(row_ids, row_sc) if iid != r and iid >= 0]
        keep = keep[:topk]
        # pad if needed
        while len(keep) < topk:
            keep.append((-1, -1.0))
        out_ids[r] = [iid for iid, _ in keep]
        out_sc[r]  = [sc  for _, sc in keep]
    return out_sc, out_ids

# ---------- main pipeline ----------
def main():
    print("[clip] starting CLIP open-vocab enrichment with ANN…")
    _ensure_dir(OUT_DIR)
    print("[clip] loading items with local images…")
    if not MAKE_CLIP or not HAS_CLIP:
        print("[skip] CLIP disabled or not available (need open_clip + pillow).")
        return

    # 1) Load items with usable local image paths
    items = load_items_with_images()
    print(f"[clip] found {len(items):,} items with local images.")
    items["title_clean"] = items["title"].map(norm_text)
    
    audit_csv = os.path.join(OUT_DIR, "items_base.csv")
    items[["i","asin","title","image_path"]].to_csv(audit_csv, index=False)
    print(f"[OK] audit CSV → {audit_csv}  (rows={len(items):,})")

    # 2) Build CLIP
    model, preprocess, device = build_clip()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    # 3) Image embeddings (mmap to control RAM)
    img_mmap = os.path.join(OUT_DIR, "clip_item_emb.mmap")
    print("[clip] encoding images (mmap)…")
    img_emb = encode_images(
        items["image_path"].tolist(),
        preprocess, model, device,
        batch_size=BATCH_IMG, mmap_path=img_mmap, dim_hint=model.visual.output_dim
    )
    # If returned memmap, ensure on-disk npy for portability
    np.save(os.path.join(OUT_DIR, "clip_item_emb.npy"), np.asarray(img_emb, dtype=np.float32))

    # 4) Title embeddings
    print("[clip] encoding titles…")
    title_emb = encode_texts(items["title_clean"].tolist(), tokenizer, model, device)
    np.save(os.path.join(OUT_DIR, "clip_title_emb.npy"), title_emb.astype("float32"))

    # 5) Alignment & simple quality score
    align = (img_emb * title_emb).sum(-1).astype("float32")

    qual_emb = encode_texts(QUALITY_HEADS, tokenizer, model, device)
    qual_sims = img_emb @ qual_emb.T  # [N, 2]
    q = qual_sims[:, 0] - qual_sims[:, 1]
    q = (q - q.min()) / (q.max() - q.min() + 1e-8)
    q = q.astype("float32")

    # 6) Concept bank
    print("[clip] mining concept bank…")
    vocab = mine_concepts(items["title_clean"].tolist(), max_vocab=20_000, min_df=5)
    vocab = list(dict.fromkeys(vocab + SEED_CONCEPTS))
    with open(os.path.join(OUT_DIR, "concept_bank.json"), "w", encoding="utf-8") as f:
        json.dump({"concepts": vocab}, f)

    print("[clip] encoding concept prompts…")
    templates = [
        "a product that is {}",
        "a close-up photo of {}",
        "consumer good, {}",
        "{}",
    ]
    prompt_texts = [tpl.format(c) for c in vocab for tpl in templates]
    concept_emb_all = encode_texts(prompt_texts, tokenizer, model, device, batch_size=BATCH_TXT)
    C, T, D = len(vocab), len(templates), concept_emb_all.shape[-1]
    concept_emb = concept_emb_all.reshape(C, T, D).mean(axis=1).astype("float32")  # [C, D]
    np.save(os.path.join(OUT_DIR, "concept_text_emb.npy"), concept_emb)

    # 7) Top-K concept tags using batched matmul (no huge S in RAM)
    print("[clip] scoring concepts per image (batched)…")
    topv = np.empty((img_emb.shape[0], TOPK_TAGS), dtype=np.float32)
    topi = np.empty((img_emb.shape[0], TOPK_TAGS), dtype=np.int32)
    for s in range(0, img_emb.shape[0], BATCH_SCORE):
        e = min(s + BATCH_SCORE, img_emb.shape[0])
        sims = img_emb[s:e] @ concept_emb.T  # [b, C]
        k = min(TOPK_TAGS, concept_emb.shape[0])
        idx = np.argpartition(-sims, kth=k-1, axis=1)[:, :k]
        vals = np.take_along_axis(sims, idx, axis=1)
        order = np.argsort(-vals, axis=1)
        topv[s:e] = np.take_along_axis(vals, order, axis=1)
        topi[s:e] = np.take_along_axis(idx,  order, axis=1)

    # diversity from softmax over topK
    # (use log2 for normalized entropy if you prefer; here we normalize by log(K))
    def _entropy_row(p):
        p = _np.maximum(p, 1e-12)
        return float(-_np.sum(p * _np.log(p)) / log(len(p)))
    # softmax on each row of topv
    exp_top = np.exp(topv - topv.max(axis=1, keepdims=True))
    probs   = exp_top / (exp_top.sum(axis=1, keepdims=True) + 1e-12)
    clip_diversity = np.apply_along_axis(_entropy_row, 1, probs).astype("float32")

    # tidy tags table (parquet)
    rows = []
    for r in range(img_emb.shape[0]):
        for k in range(topv.shape[1]):
            rows.append({
                "i": int(items.at[r, "i"]),
                "asin": items.at[r, "asin"],
                "concept": vocab[int(topi[r, k])],
                "score": float(topv[r, k]),
                "rank": int(k+1),
            })
    tags_df = pd.DataFrame(rows)
    tags_path = os.path.join(OUT_DIR, "topk_tags.parquet")
    tags_df.to_parquet(tags_path, index=False)
    print(f"[OK] tags → {tags_path} (rows={len(tags_df):,})")

    # 8) ANN neighbors with FAISS (no NxN)
    print("[ann] building FAISS IVF+PQ for image neighbors…")
    if not HAS_FAISS:
        raise RuntimeError("faiss not installed; install faiss-gpu or faiss-cpu to enable ANN neighbors.")
    index, is_gpu = build_faiss_index(
        x=np.asarray(img_emb, dtype=np.float32),
        use_gpu=FAISS_USE_GPU,
        nlist=IVF_NLIST, m=PQ_M, train_max=IVF_TRAIN_MAX, nprobe=IVF_PROBE
    )
    print(f"[ann] index ready. gpu={is_gpu}  nlist={IVF_NLIST}  m={PQ_M}  nprobe={IVF_PROBE}")

    nn_vals, nn_ids = ann_self_search(index, np.asarray(img_emb, dtype=np.float32), topk=TOPK_NEIGHBORS)
    # write neighbors as parquet (sparse rows)
    nn_rows = []
    for r in range(img_emb.shape[0]):
        for k in range(nn_ids.shape[1]):
            nid = int(nn_ids[r, k])
            if nid < 0: continue
            nn_rows.append({
                "i": int(items.at[r, "i"]),
                "asin": items.at[r, "asin"],
                "neighbor_i": int(items.at[nid, "i"]),
                "neighbor_asin": items.at[nid, "asin"],
                "sim": float(nn_vals[r, k]),
                "rank": int(k+1),
            })
    nn_df = pd.DataFrame(nn_rows)
    nn_path = os.path.join(OUT_DIR, "nn_ann.parquet")
    nn_df.to_parquet(nn_path, index=False)
    print(f"[OK] ANN neighbors → {nn_path} (rows={len(nn_df):,})")

    # 9) Compact per-item features (for wide table)
    topk_concepts_json = []
    for r in range(img_emb.shape[0]):
        tags = [{"c": str(vocab[int(topi[r, k])]), "s": float(topv[r, k])} for k in range(topv.shape[1])]
        topk_concepts_json.append(json.dumps(tags, ensure_ascii=False))

    feat_df = pd.DataFrame({
        "i": items["i"].astype(int),
        "asin": items["asin"].astype(str),
        "clip_title_align": align.astype("float32"),
        "clip_quality_score": q.astype("float32"),
        "clip_diversity": clip_diversity,
        "topk_concepts": topk_concepts_json,
    })
    feat_path = os.path.join(OUT_DIR, "clip_features.parquet")
    feat_df.to_parquet(feat_path, index=False)
    print(f"[OK] features → {feat_path}")

    # metadata for reproducibility
    with open(os.path.join(OUT_DIR, "clip_item_order.json"), "w", encoding="utf-8") as f:
        json.dump({
            "i_order": items["i"].astype(int).tolist(),
            "asin_order": items["asin"].astype(str).tolist(),
            "image_path": items["image_path"].astype(str).tolist(),
        }, f)
    print("[DONE] CLIP enrichment with ANN complete.")

if __name__ == "__main__":
    main()
