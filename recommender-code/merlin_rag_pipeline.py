# merlin_rag_pipeline.py
import nvtabular as nvt
from nvtabular.ops import LambdaOp, Categorify, Normalize
from merlin.schema.tags import Tags
from merlin.schema import Schema
import glob

# -------- Paths --------
TRAIN_PATH = "data/train/*.parquet"   # your parquet shards
VALID_PATH = "data/valid/*.parquet"
OUTPUT_PATH = "proc/"

# -------- Column sets --------
USER_ID = ["user_id"]
ITEM_ID = ["item_id"]
LABEL = ["label"]
TS = ["ts"]

# ITEM SIDE: Vision-RAG
ITEM_IMAGE_EMB = ["item_image_emb"]        # fixed-size float vector (shape [D])
ITEM_RAG_TAGS = ["item_rag_tags"]          # list[str] tags

# OPTIONAL context (example)
CONTEXT_CONT = []                           # e.g., ["price"]
CONTEXT_CAT = []                            # e.g., ["country", "device"]

# -------- Workflows --------
# Categorical
cat_user = USER_ID >> Categorify()
cat_item = ITEM_ID >> Categorify()
cat_rag  = ITEM_RAG_TAGS >> Categorify()   # handles list[str] -> list[int] vocab

# Continuous
cont_ctx = CONTEXT_CONT >> Normalize()

# Image embedding: pass-through (already numeric). Ensure it's a list/array of floats per row.
# If stored as a Python list in parquet, NVTabular will treat it as a list column.
img_pass = ITEM_IMAGE_EMB >> LambdaOp(lambda col: col)  # no-op to keep in workflow

# Build workflow graph
wf = nvt.Workflow(cat_user + cat_item + cat_rag + cont_ctx + img_pass + LABEL + TS)

# Fit / transform
train_ds = nvt.Dataset(glob.glob(TRAIN_PATH))
valid_ds = nvt.Dataset(glob.glob(VALID_PATH))

wf.fit(train_ds)
proc_train = wf.transform(train_ds)
proc_valid = wf.transform(valid_ds)

proc_train.to_parquet(output_path=f"{OUTPUT_PATH}/train")
proc_valid.to_parquet(output_path=f"{OUTPUT_PATH}/valid")

# --------- Tag the schema so Merlin knows what's what ----------
schema: Schema = wf.output_schema

schema = schema.with_column(
    schema["user_id"].with_tags([Tags.USER_ID, Tags.CATEGORICAL])
)
schema = schema.with_column(
    schema["item_id"].with_tags([Tags.ITEM_ID, Tags.CATEGORICAL])
)
schema = schema.with_column(
    schema["label"].with_tags([Tags.BINARY_CLASSIFICATION, Tags.TARGET])
)
schema = schema.with_column(
    schema["ts"].with_tags([Tags.TIMESTAMP, Tags.CONTEXT])
)

# Item Vision-RAG columns:
# item_image_emb: a list/array of floats – tag as ITEM + CONTINUOUS + EMBEDDING
schema = schema.with_column(
    schema["item_image_emb"].with_tags([Tags.ITEM, Tags.CONTINUOUS, Tags.EMBEDDING])
)
# item_rag_tags: list categorical – tag as ITEM + CATEGORICAL + LIST
schema = schema.with_column(
    schema["item_rag_tags"].with_tags([Tags.ITEM, Tags.CATEGORICAL, Tags.LIST])
)

# Optional context columns tagging
for c in CONTEXT_CAT:
    schema = schema.with_column(schema[c].with_tags([Tags.CONTEXT, Tags.CATEGORICAL]))
for c in CONTEXT_CONT:
    schema = schema.with_column(schema[c].with_tags([Tags.CONTEXT, Tags.CONTINUOUS]))

# Save schema
schema_path = f"{OUTPUT_PATH}/schema.json"
schema.to_json(schema_path)
print("Saved processed data + schema at:", OUTPUT_PATH)
