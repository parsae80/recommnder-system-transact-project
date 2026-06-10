# Amazon Personal Care Recommendation Pipeline

This repository contains the scripts used to build the two-stage recommendation pipeline:

1. preprocess item metadata and reviews,
2. generate CLIP features and top-k concepts,
3. train LightGCN / feature-aware LGCN,
4. train the second-stage TransAct model on top of the learned item embeddings.

The codebase is organized around the actual scripts in this folder, not around a package or library layout.

## Main Scripts

- `prep_items_meta.py`  
  Builds item metadata features from Amazon meta JSONL and writes `items_features.parquet`, `item_id_map.parquet`, and optionally `cobuy_edges.parquet`.

- `prep_reviews_revised.py`  
  Converts the reviews JSONL into `interactions.parquet` and `reviews_agg.parquet`.

- `media_downloader.py`  
  Downloads product images from the item metadata and writes the image manifest files.

- `clip_feature_gen.py`  
  Generates CLIP embeddings, concept bank files, top-k concept parquet, and ANN-related artifacts.

- `clip_feature_gen_paths.py`  
  Variant of the CLIP generator with the same pipeline but path overrides tuned for the handmade dataset layout.

- `Untitled-1.py`  
  Main LightGCN / feature-aware LGCN training script. It consumes the CLIP artifacts when `use_clip_dense` is enabled.

- `tomp.py` and `tomp_baseline_30m.py`  
  TransAct second-stage scripts. They load the pretrained item embeddings from stage 1 and feed them into the transformer model.

## Data Flow

The pipeline is roughly:

1. build item features with `prep_items_meta.py`
2. build review interactions with `prep_reviews_revised.py`
3. download images with `media_downloader.py`
4. generate CLIP embeddings and top-k concepts with `clip_feature_gen.py` or `clip_feature_gen_paths.py`
5. train LightGCN / feature-aware LGCN with `Untitled-1.py`
6. train the second-stage TransAct model with `tomp.py`

## Key Outputs

Typical output locations are:

- `amazon_out/items_meta/items_features.parquet`
- `amazon_out/items_meta/item_id_map.parquet`
- `amazon_out/reviews_revised_proc/interactions.parquet`
- `amazon_out/reviews_revised_proc/reviews_agg.parquet`
- `amazon_handmade_out/downloaded_media_all/images_per_asin.jsonl`
- `amazon_handmade_out/downloaded_media_all/downloads_manifest.parquet`
- `amazon_handmade_out/clip/clip_item_emb.npy`
- `amazon_handmade_out/clip/topk_tags.parquet`
- `item_tower/*.pt` for the trained LightGCN exports

## CLIP Artifacts

The CLIP generator writes the artifacts consumed by the stage-1 LightGCN script:

- item embeddings: `clip_item_emb.npy`
- top-k concept table: `topk_tags.parquet`
- concept embedding bank: `concept_text_emb.npy`
- concept vocabulary: `concept_bank.json`

The exact output paths depend on whether you use `clip_feature_gen.py` or `clip_feature_gen_paths.py`.


## Environment

Recommended packages:

- `pandas`
- `numpy`
- `pyarrow`
- `torch`
- `open_clip_torch`
- `pillow`
- `faiss` or `faiss-gpu` if you want ANN neighbors in the CLIP script

Example CPU setup:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pandas numpy pyarrow torch open_clip_torch pillow
```

Example GPU setup:

```bash
conda create -y -n recsys python=3.10
conda activate recsys
conda install -y pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
pip install pandas numpy pyarrow open_clip_torch pillow faiss-gpu
```

## Typical Run Order

```bash
python prep_items_meta.py
python prep_reviews_revised.py
python media_downloader.py
python clip_feature_gen.py
python lgcn file
python transact file 
```

## Notes

- The repo currently includes several experiment and baseline scripts. Not every file is part of the main pipeline.
- The CLIP scripts can run in a CPU-only mode, but GPU is strongly preferred.
- If you only want the core pipeline, the five scripts above are the ones to keep track of.
