# Vision-RAG for Amazon 2023 — Project README
.

---

## TL;DR

- **Goal:** Predict/complete product metadata (e.g., `brand`, later `finish/coverage/size`) using **Clip** and compare against simple baseline ().
- **Datasets:** Amazon Reviews 2023 (category: **All_Beauty**) — reviews & meta.
- **Key outputs:**
  - Catalog: `/home/p_esla/rec_proj/amazon_out/items_meta.parquet`
  - RAG jobs: `/home/p_esla/rec_proj/amazon_out/rag_eval/vision_rag_jobs.jsonl`
  - RAG truth: `/home/p_esla/rec_proj/amazon_out/rag_eval/truth.parquet`
  - Media downloads: `/home/p_esla/rec_proj/amazon_out/media_all/...`
  - Interactions (labels/weights): `/home/p_esla/rec_proj/amazon_out/interactions.parquet`
  - CLIP enrichment: `/home/p_esla/rec_proj/amazon_out/clip/...`

---

## Data Sources

### Raw inputs (Amazon 2023)
- **Reviews (All_Beauty)**  
  `"/home/p_esla/rec_proj/am_dataset/amazon2023/All_Beauty.jsonl.gz"`  
  Example schema (per line):
  ```json
  {"rating":5.0,"title":"Such a lovely scent...","text":"...","images":[],"asin":"B00YQ6X8EO","parent_asin":"B00YQ6X8EO","user_id":"...","timestamp":1588687728923,"helpful_vote":0,"verified_purchase":true}
  ```
- **Metadata (All_Beauty)**  
  `"/home/p_esla/rec_proj/am_dataset/amazon_meta2023/meta_All_Beauty.jsonl.gz"`  
  Example schema (per line):
  ```json
  {"main_category":"All Beauty","title":"Howard ...","average_rating":4.8,"rating_number":10,
   "features":[],"description":[],"price":null,"images":[{"thumb":"...","large":"...","variant":"MAIN"}],
   "videos":[],"store":"Howard Products","categories":[],"details":{"UPC":"..."}, "parent_asin":"B01CUPMQZE"}
  ```

> Reviews give **`asin` + `parent_asin`** mapping; meta typically attaches **images/brand/category** at the **`parent_asin`** level. We bridge them.

---

## Project Layout (key files)

```
/home/p_esla/rec_proj/
├─ am_dataset/
│  ├─ amazon2023/All_Beauty.jsonl.gz
│  └─ amazon_meta2023/meta_All_Beauty.jsonl.gz
├─ amazon_out/
│  ├─ items_meta.parquet                # per-ASIN catalog with image URLs, brand, categories
│  ├─ items_meta_sample.parquet         # optional stratified sample for quick runs
│  ├─ clip/
│  │  ├─ items_base.csv                 # audit table used by rag_enrich
│  │  ├─ rag_enriched.csv               # heuristics from titles (color/material/style/price_estimate)
│  │  ├─ clip_item_emb.npy              # (optional) CLIP embeddings if CLIP enabled
│  │  └─ clip_item_order.json           # alignment helper for CLIP embeddings
│  ├─ media_all/
│  │  ├─ downloads_manifest.parquet     # per-file download status (after completion or recreated)
│  │  └─ images_per_asin.jsonl          # asin -> [local image paths]
│  ├─ rag_eval/
│  │  ├─ vision_rag_jobs.jsonl          # input jobs (can be sharded to jobs_000, ...)
│  │  ├─ truth.parquet                  # ground truth labels (e.g., asin + brand)
│  │  ├─ preds.jsonl / preds_*.jsonl    # predictions from RAG or baselines
│  │  └─ (optional) shards: jobs_000 ... jobs_011
│  ├─ interactions.parquet              # user-item labels/weights from reviews
│  └─ item_reference_reviews.parquet    # tiny audit refs (optional)
└─ rag_sec/
   ├─ prep.py / prepr.py                # main pipeline (argument-free config block)
   ├─ item_meta_prep.py                 # builds items_meta.parquet from reviews+meta
   ├─ build_rag_jobs_from_meta.py       # creates jobs + truth
   ├─ download_all_media_nocli.py       # threaded media downloader (no CLI args in your copy)
   ├─ rag_enrich.py                     # optional: heuristics + CLIP features
   ├─ clip_brand_eval.py                # CLIP zero-shot brand eval (argument-free)
   ├─ utils.py                          # review selection / general utils (trimmed)
   ├─ registry.py                       # category → requested attributes mapping
   └─ new_rag.py                        # your Vision-RAG runner (to be adapted to new artifacts)
```

---

## Pipeline Stages

### A) **Catalog / Item-meta Track (Vision inputs)**

1. **Build `items_meta.parquet`**  
   Script: `item_meta_prep.py`  
   Output: `/home/p_esla/rec_proj/amazon_out/items_meta.parquet`  
   Columns:
   ```
   asin, title, brand, category_path, image_urls, mapped_category
   ```
   Notes:
   - Join strategy: bridge `asin→parent_asin` from **reviews** to merge **meta** at `parent_asin`.
   - `mapped_category` is a normalized category (via `registry.py`).

2. **(Optional) Sample items for quick iteration**  
   Output: `/home/p_esla/rec_proj/amazon_out/items_meta_sample.parquet`

3. **Build Vision-RAG jobs & truth**  
   Script: `build_rag_jobs_from_meta.py`  
   Inputs: `items_meta.parquet`  
   Outputs:
   - `rag_eval/vision_rag_jobs.jsonl` (mask target attrs)
   - `rag_eval/truth.parquet` with columns: `asin`, `brand` (and any other target attrs you add)

4. **Download media (resume-friendly)**  
   Script: `download_all_media_nocli.py`  
   Inputs: `items_meta.parquet` (or the sample)  
   Outputs:
   - `/media_all/<shard>/<ASIN>/img_XXX.jpg`
   - `/media_all/downloads_manifest.parquet`
   - `/media_all/images_per_asin.jsonl`

5. **(Optional) CLIP enrichment / heuristics**  
   Script: `rag_enrich.py`  
   - `MAKE_CLIP=False` → only writes `items_base.csv` + `rag_enriched.csv` (title heuristics).  
   - `MAKE_CLIP=True` (with `open_clip_torch` + `pillow`) → also writes `clip_item_emb.npy` and alignment JSON.

---

### B) **Reviews / Interactions Track (Labels & weights)**

1. **Active user filtering, labeling & weights**  
   Script: `prep.py` (or `prepr.py`), argument-free config  
   Output: `/home/p_esla/rec_proj/amazon_out/interactions.parquet`  
   Columns:
   ```
   user_id, asin, rating, verified, helpful_vote, timestamp, label, weight
   ```
   - Labels: +1 (verified & rating ≥ 4), −1 (verified & rating ≤ 2), else 0.
   - Weights: verified + helpfulness + recency decay.

2. **Item reference reviews** (tiny audit set, optional)  
   Output: `/home/p_esla/rec_proj/amazon_out/item_reference_reviews.parquet`

> You can later map `asin → i` to create `item_features.parquet` for CTR training.

---

### C) **Vision-RAG Inference & Evaluation**

1. **Run Vision-RAG**  
   - Input: `rag_eval/vision_rag_jobs.jsonl` (or shards `jobs_000..`)  
   - Output: `rag_eval/preds.jsonl` with lines like:
     ```json
     {"asin":"B00YQ6X8EO","attrs":{"brand":"Howard Products"},"conf":{"brand":0.92}}
     ```

2. **Score**  
   - Compare `preds.jsonl` vs `truth.parquet`  
   - Metrics: exact/loose match, coverage.

---

## Baselines & Offline Testing

When networking/CLIP aren’t available, you can still exercise the pipeline:

- **Echo truth** stub → `preds_echo.jsonl` (pipeline smoke test; ~100% expected).
- **Title-only heuristic** stub → `preds_title_heur.jsonl` (no images).
- **Random (brand freq)** stub → `preds_rand.jsonl` (sanity baseline).
- **CLIP zero-shot brand** (image-only) → `clip_brand_eval.py`  
  - Inputs: `truth.parquet`, `media_all/images_per_asin.jsonl`  
  - Output: `preds_clip.jsonl` + printed metrics.

---

## Configuration Knobs (common)

- **Sampling (for speed)**
  - `items_meta_sample.parquet` (pre-sampled file)  
  - Or knobs **inside** scripts (e.g., `SAMPLE_N`, `N_PER_CAT`, etc.)

- **rag_enrich.py**
  - `MAKE_CLIP = True/False`
  - `DOWNLOAD_IMAGES = True/False` (if present in your local copy)
  - Device auto-select (CUDA if available)

- **prep.py (reviews → interactions)**
  - `ACTIVE_WINDOW_DAYS`
  - `MIN_VERIFIED_IN_WINDOW`
  - `MIN_VERIFIED_RATIO`
  - `MIN_MEDIAN_HELPFUL`

- **build_rag_jobs_from_meta.py**
  - `TARGET_ATTRS = ["brand"]` (extend later: `"finish"`, `"coverage"`, `"size_ml"`, …)

---

## Typical Commands

**Peek head of gz JSONL (reviews):**
```bash
zcat /home/p_esla/rec_proj/am_dataset/amazon2023/All_Beauty.jsonl.gz | head -n 3
```

**Build items meta:**
```bash
python /home/p_esla/rec_proj/rag_sec/item_meta_prep.py
```

**Create RAG jobs + truth (brand):**
```bash
python /home/p_esla/rec_proj/rag_sec/build_rag_jobs_from_meta.py
```

**Shard jobs (optional):**
```bash
cd /home/p_esla/rec_proj/amazon_out/rag_eval
split -d -a 3 -l 10000 vision_rag_jobs.jsonl jobs_
```

**Download media (no-CLI script):**
```bash
python /home/p_esla/rec_proj/rag_sec/download_all_media_nocli.py
```

**Run CLIP zero-shot brand eval:**
```bash
python /home/p_esla/rec_proj/rag_sec/clip_brand_eval.py
```

---

## Environment

**Optional isolated venv (CPU):**
```bash
python3 -m venv /home/p_esla/rec_proj/venvs/visionrag
source /home/p_esla/rec_proj/venvs/visionrag/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install open_clip_torch pillow pandas pyarrow numpy
# deactivate with:  deactivate
```

**Conda env (GPU):**
```bash
conda create -y -n visionrag python=3.10
conda activate visionrag
conda install -y pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
pip install open_clip_torch pillow pandas pyarrow numpy
# deactivate with:  conda deactivate
```

---

## Troubleshooting Notes

- **“[skip] CLIP not available”** → Not an error; it just skipped embeddings. Install `open_clip_torch` + `pillow` (+ Torch) to enable.
- **Network/DNS issues (conda/pip)** → Your box had DNS failures. Fix `/etc/resolv.conf` or set proxies; meanwhile you can proceed with `MAKE_CLIP=False`.
- **Stopping media download mid-run** → Safe. Files are persisted per image. Next run **resumes** and skips completed files. Manifest can be rebuilt from disk if needed (you have a snippet).
- **Series not JSON-serializable** (when writing jobs) → ensure you convert pandas objects to Python lists/strings (your builder was revised to handle this).

---

## What’s Evaluated Today

- **Primary target:** `brand` (from meta)  
  - Jobs **mask** brand; truth keeps it for scoring.
- **Planned targets:** category-specific attributes (e.g., `finish`, `coverage_level`, `size_ml`, `shade`) — only where **vision-based** extraction is plausible and valuable.

---

## Next Steps (suggested)

1. Add **prompt-ensembles** to CLIP text (e.g., `“the {brand} logo”, “a product made by {brand}”`) and average text embeddings.
2. Add **OCR-assisted Vision-RAG** (logo/packaging text → retrieval → attribute extraction).
3. Build a **multimodal supervised baseline**:
   - Text TF-IDF (title) + CLIP image emb → small MLP → predict brand/finish.
4. Expand **TARGET_ATTRS** by **category**, only where images can plausibly reveal them.
5. Create **item_features.parquet**:
   - Merge Vision-RAG attrs + CLIP vectors + title heuristics → feed into CTR model with `interactions.parquet`.

---

## Provenance / Audit

- Items with images: **115,709 / 115,709** (after fixing `image_urls` normalization).
- RAG jobs & truth were re-generated to non-zero rows once `image_urls` became real `list[str]`.
- Media download reached **18k+ images** mid-run; resumable and persisted.
- CLIP step was initially skipped due to missing `open_clip_torch`/network.
