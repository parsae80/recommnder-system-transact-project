#!/usr/bin/env python3
# prep_reviews.py

from pathlib import Path
import numpy as np
import pandas as pd

# ========= EDIT THESE =========from pathlib import Path

# If this file is inside src/, use parents[1].
# If this file is in the repo root, use .parent.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_ROOT = PROJECT_ROOT / "new_amazon_dataset"
DATA_ROOT = PROJECT_ROOT / "amazon_personal_care_out"

REVIEWS_JSONL = RAW_DATA_ROOT / "Beauty_and_Personal_Care.jsonl.gz"

ITEM_IDMAP_PARQUET = DATA_ROOT / "items_meta" / "item_id_map.parquet"

OUT_DIR = DATA_ROOT / "reviews_revised_proc"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INTER_PARQUET = OUT_DIR / "interactions.parquet"
AGG_PARQUET = OUT_DIR / "reviews_agg.parquet"
# ==============================

def main():
    idmap = pd.read_parquet(ITEM_IDMAP_PARQUET)
    parent2i = idmap.set_index("parent_asin")["i"].to_dict()

    CHUNK = 300_000
    inter_parts = []
    aggs = []

    # Use only essential columns
    use_cols = ["parent_asin", "user_id", "rating", "timestamp", "verified_purchase"]
    reader = pd.read_json(REVIEWS_JSONL, lines=True, chunksize=CHUNK)

    user2u = {}; next_u = 0

    for ch in reader:
        for c in use_cols:
            if c not in ch.columns: ch[c] = np.nan

        ch["parent_asin"] = ch["parent_asin"].astype(str)
        ch["rating"] = pd.to_numeric(ch["rating"], errors="coerce")
        ch["timestamp"] = pd.to_numeric(ch["timestamp"], errors="coerce").fillna(0).astype("int64") // 1000
        
        # Convert bool to int (0/1) for PyTorch compatibility
        ch["verified_purchase"] = ch["verified_purchase"].fillna(False).astype(int)
        
        ch = ch.dropna(subset=["parent_asin","user_id","rating"])
        ch["i"] = ch["parent_asin"].map(parent2i)
        ch = ch.dropna(subset=["i"])
        ch["i"] = ch["i"].astype("int32")

        new_users = ~ch["user_id"].isin(user2u)
        if new_users.any():
            for uid in ch.loc[new_users, "user_id"].unique():
                user2u[uid] = next_u; next_u += 1
        ch["u"] = ch["user_id"].map(user2u).astype("int32")

        # Label: 1 if high satisfaction, 0 otherwise
        ch["label"] = (ch["rating"] >= 4.0).astype("int8")

        # Interaction file for the Deep Path (Transformer)
        inter_parts.append(ch[["u","i","timestamp","label", "verified_purchase"]].rename(columns={"timestamp":"time_stamp"}))

        # Per-chunk aggregates for the Wide Path
        grp = ch.groupby("i", as_index=False).agg(
            rev_count=("rating","size"),
            rev_rating_sum=("rating","sum"), 
            rev_verified_sum=("verified_purchase","sum"),
        )
        aggs.append(grp)

    # Save Interactions
    inter = pd.concat(inter_parts, ignore_index=True) if inter_parts else pd.DataFrame()
    inter = inter.sort_values(["u","time_stamp"]).reset_index(drop=True)
    inter.to_parquet(INTER_PARQUET, index=False)

    # Final Merge for Item Features
    if aggs:
        agg = pd.concat(aggs, ignore_index=True).groupby("i", as_index=False).agg(
            rev_count=("rev_count","sum"),
            rev_rating_sum=("rev_rating_sum","sum"),
            rev_verified_sum=("rev_verified_sum","sum"),
        )
        
        # Final wide features
        agg["rev_rating_avg"] = (agg["rev_rating_sum"] / agg["rev_count"]).astype("float32")
        agg["rev_verified_ratio"] = (agg["rev_verified_sum"] / agg["rev_count"]).astype("float32")
        
        agg = agg.drop(columns=["rev_rating_sum"])
        agg.to_parquet(AGG_PARQUET, index=False)
        print(f"[OK] Created {AGG_PARQUET} with verified ratio.")
    else:
        print("[warn] No data processed.")

if __name__ == "__main__":
    main()
