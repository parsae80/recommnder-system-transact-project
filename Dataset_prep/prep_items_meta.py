#!/usr/bin/env python3
# prep_items_meta.py
# Input: meta JSONL (.gz) with columns like:
#   ['average_rating','rating_number','price','images','store','main_category',
#    'title','parent_asin','bought_together', ...]
# Output:
#   - items_features.parquet  (one row per parent_asin, with handy features)
#   - item_id_map.parquet     (contiguous i -> parent_asin)
#   - cobuy_edges.parquet     (optional item-item edges A,B,weight)

# Quick shell preview examples (replace with your file path):
# View keys of the first record:
#   zcat /path/to/meta.jsonl.gz | head -n 1 | jq 'keys'
# View the full first record:
#   zcat /path/to/meta.jsonl.gz | head -n 1 | jq .

import re, math, json
from pathlib import Path
from typing import Any, List
import numpy as np
import pandas as pd
# View the keys of the first record only
# If this file is inside src/, use parents[1].
# If this file is in the repo root, use .parent.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "am_dataset"
OUT_ROOT = PROJECT_ROOT / "amazon_out"

ITEMS_JSONL = DATA_ROOT / "amazon_meta2023" / "meta_All_Beauty.jsonl.gz"

OUT_DIR = OUT_ROOT / "items_meta"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ITEMS_PARQUET = OUT_DIR / "items_features.parquet"
IDMAP_PARQUET = OUT_DIR / "item_id_map.parquet"
COBUY_PARQUET = OUT_DIR / "cobuy_edges.parquet"  # optional edges
TOP_K_IMAGES  = 8
MIN_COBUY_SUPPORT = 2
# ==============================

def _pick_image_urls(imgs: Any, k=TOP_K_IMAGES) -> List[str]:
    if not isinstance(imgs, list): return []
    out, seen = [], set()
    for d in imgs:
        if not isinstance(d, dict): continue
        u = d.get("hi_res") or d.get("large") or d.get("thumb")
        if isinstance(u, str) and u and u not in seen:
            seen.add(u); out.append(u)
        if len(out) >= k: break
    return out

def _price_to_float(x):
    if x is None or (isinstance(x, float) and math.isnan(x)): return np.nan
    s = re.sub(r"[^\d.,]", "", str(x))
    s = s.replace(",", "")
    try: return float(s) if s else np.nan
    except: return np.nan

def main():
    meta = pd.read_json(ITEMS_JSONL, lines=True, compression="gzip")
    keep = ["parent_asin","title","store","main_category","images",
            "average_rating","rating_number","price","bought_together"]
    for k in keep:
        if k not in meta.columns: meta[k] = np.nan
    df = meta[keep].copy()

    # Basic item table
    df["parent_asin"] = df["parent_asin"].astype(str)
    df["brand"] = df["store"].astype(str).fillna("")
    df["category_path"] = df["main_category"].astype(str).fillna("")
    df["image_urls"] = df["images"].apply(_pick_image_urls)
    df["img_count"] = df["images"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df["avg_rating_meta"] = pd.to_numeric(df["average_rating"], errors="coerce")
    df["rating_count_meta"] = pd.to_numeric(df["rating_number"], errors="coerce").fillna(0).astype("int64")
    df["price_num"] = df["price"].apply(_price_to_float)
    df["title"] = df["title"].fillna("").astype(str)

    # Stable ID i
    items = df.drop_duplicates(subset=["parent_asin"]).copy()
    items = items.sort_values("parent_asin").reset_index(drop=True)
    items["i"] = np.arange(len(items), dtype=np.int32)

    # Convenience features for item tower
    items["has_images"] = (items["img_count"] > 0).astype("int8")
    items["pop_log"] = np.log1p(items["rating_count_meta"].astype("float32"))

    items_out = items[[
        "i","parent_asin","title","brand","category_path",
        "image_urls","img_count","has_images",
        "avg_rating_meta","rating_count_meta","price_num","pop_log"
    ]]

    items_out.to_parquet(ITEMS_PARQUET, index=False)
    items_out[["i","parent_asin"]].to_parquet(IDMAP_PARQUET, index=False)
    print(f"[OK] items → {ITEMS_PARQUET} rows={len(items_out):,}")
    print(f"[OK] idmap → {IDMAP_PARQUET}")

    # ---------- Optional: co-buy edges from 'bought_together' ----------
    # Many dumps store bought_together as a list of ASINs related to the *parent_asin* row.
    if "bought_together" in df.columns:
        bt = df[["parent_asin","bought_together"]].copy()
        # explode lists → (A, B)
        bt = bt[bt["bought_together"].apply(lambda v: isinstance(v, list))]
        if bt.empty:
            print("[WARN] no bought_together lists found, skipping cobuy edges")
            return
        bt = bt.explode("bought_together").dropna()
        bt["bought_together"] = bt["bought_together"].astype(str)

        # Map both endpoints to parent_asin space when possible (here we assume the IDs are parent_asin-like;
        # if they are child ASINs, you’d need a child→parent map first).
        # Self-edges removed; count support
        edges = (bt[bt["parent_asin"] != bt["bought_together"]]
                 .groupby(["parent_asin","bought_together"], as_index=False)
                 .size())
        edges.rename(columns={"size":"weight"}, inplace=True)
        edges = edges[edges["weight"] >= MIN_COBUY_SUPPORT]

        # Map to i indices (filter to items we know)
        amap = items_out.set_index("parent_asin")["i"].to_dict()
        edges["ia"] = edges["parent_asin"].map(amap)
        edges["ib"] = edges["bought_together"].map(amap)
        edges = edges.dropna(subset=["ia","ib"]).astype({"ia":"int32","ib":"int32","weight":"int32"})

        # Symmetrize (optional but common for LightGCN): duplicate reversed edge
        edges_sym = pd.concat([edges[["ia","ib","weight"]],
                               edges.rename(columns={"ia":"ib","ib":"ia"})[["ia","ib","weight"]]],
                              ignore_index=True).drop_duplicates(subset=["ia","ib"])

        edges_sym.to_parquet(COBUY_PARQUET, index=False)
        print(f"[OK] cobuy edges → {COBUY_PARQUET} rows={len(edges_sym):,} (min_support={MIN_COBUY_SUPPORT})")

if __name__ == "__main__":
    main()
