# train_ttw_rag.py
import torch
from torch.optim import Adam
from merlin.schema import Schema
from merlin.io import Dataset
from merlin.dataloader.torch import Loader
from merlin.schema.tags import Tags

from merlin.models.torch import Model as MModel
from merlin.models.torch import blocks, retrieval, utils

# -------- Config: turn features on/off for ablations --------
USE_IMAGE_EMB = True      # set False to disable image embeddings
USE_RAG_TAGS  = True      # set False to disable rag tags
EMB_DIM = 64              # ID/tag embedding dim
IMG_PROJ_DIM = 64         # projection dim for image emb (if D is large)
LR = 1e-3
BATCH = 2048
EPOCHS = 3

PROC_TRAIN = "proc/train"
PROC_VALID = "proc/valid"
SCHEMA_PATH = "proc/schema.json"

# -------- Load schema and filter features --------
schema = Schema().from_json(SCHEMA_PATH)

user_schema = schema.select_by_tag(Tags.USER_ID) + schema.select_by_tag(Tags.CONTEXT)
item_schema = schema.select_by_tag(Tags.ITEM_ID)

# Append Vision-RAG item side if enabled
if USE_IMAGE_EMB:
    item_schema = item_schema + schema.select_by_name(["item_image_emb"])
if USE_RAG_TAGS:
    item_schema = item_schema + schema.select_by_name(["item_rag_tags"])

target_schema = schema.select_by_tag(Tags.TARGET)

# -------- DataLoaders --------
train_ds = Dataset(PROC_TRAIN, engine="parquet")
valid_ds = Dataset(PROC_VALID, engine="parquet")

train_loader = Loader(train_ds, batch_size=BATCH, shuffle=True, schema=(user_schema + item_schema + target_schema))
valid_loader = Loader(valid_ds, batch_size=BATCH, shuffle=False, schema=(user_schema + item_schema + target_schema))

# -------- Build towers --------
# Embedding + MLP blocks are auto-built from schema by Merlin Models blocks
user_inputs = blocks.InputBlock(user_schema)
user_embs   = blocks.EmbeddingFeatures(
    user_schema.select_by_tag(Tags.CATEGORICAL), embedding_dim=EMB_DIM
)
user_cont   = blocks.ContinuousFeatures(user_schema.select_by_tag(Tags.CONTINUOUS))
user_body   = blocks.ConcatBlock([user_embs, user_cont]) if len(user_cont.features) > 0 else user_embs
user_tower  = blocks.MLPBlock([128, EMB_DIM], activation="relu")(user_body)

# Item ID embedding
item_inputs = blocks.InputBlock(item_schema)
item_id_emb = blocks.EmbeddingFeatures(
    item_schema.select_by_tag(Tags.CATEGORICAL).select_by_name(["item_id"]), embedding_dim=EMB_DIM
)

# Item RAG TAGS: list-categorical -> pooled embedding
item_tag_block = None
if USE_RAG_TAGS and "item_rag_tags" in [c.name for c in item_schema]:
    item_tag_block = blocks.EmbeddingFeatures(
        item_schema.select_by_name(["item_rag_tags"]), embedding_dim=EMB_DIM, 
        pooling="mean"  # mean/max/sum; mean is usually robust
    )

# Item IMAGE EMB: continuous vector -> small projection/gating
item_img_block = None
if USE_IMAGE_EMB and "item_image_emb" in [c.name for c in item_schema]:
    # Take the raw continuous vector and project to IMG_PROJ_DIM
    cont_block = blocks.ContinuousFeatures(item_schema.select_by_name(["item_image_emb"]))
    item_img_block = blocks.MLPBlock([256, IMG_PROJ_DIM], activation="relu")(cont_block)

# Concat item features
item_feats = [item_id_emb]
if item_tag_block is not None:
    item_feats.append(item_tag_block)
if item_img_block is not None:
    item_feats.append(item_img_block)

item_body = blocks.ConcatBlock(item_feats) if len(item_feats) > 1 else item_feats[0]

# OPTIONAL: learnable gate to balance ID vs image/text (simple)
if (item_img_block is not None) or (item_tag_block is not None):
    item_body = blocks.MLPBlock([128, EMB_DIM], activation="relu")(item_body)
else:
    # Already at EMB_DIM from ID emb
    pass

item_tower = item_body  # final item vector

# -------- Retrieval task (in-batch negatives) --------
task = retrieval.RecallAtKTask(top_k=50)  # for metrics only

model = retrieval.TwoTowerModel(
    user_tower=user_tower,
    item_tower=item_tower,
    query_inputs=user_inputs,
    item_inputs=item_inputs,
    temperature=1.0,                 # in-batch softmax temperature
    log_qk=True,                      # optional logging
    loss=utils.ContrastiveLoss(),     # in-batch negative sampling
    metrics=[task],
)

opt = Adam(model.parameters(), lr=LR)

# -------- Training loop --------
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        for k in batch:
            batch[k] = batch[k].to(device)
        loss_dict = model.training_step(batch)
        loss = loss_dict["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    avg = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}: train loss {avg:.4f}")

    # Validation (Recall@K)
    model.eval()
    with torch.no_grad():
        metric_accum = []
        for batch in valid_loader:
            for k in batch:
                batch[k] = batch[k].to(device)
            _ = model.validation_step(batch)  # merlin-models updates internal metrics
        metrics = model.compute_metrics()
        print(f"Valid metrics: {metrics}")
        model.reset_metrics()
