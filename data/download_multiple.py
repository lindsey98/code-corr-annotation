# pip install datasets
import os
import gzip, shutil
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset

# McEval
ds_mceval = load_dataset(
    "Multilingual-Multimodal-NLP/McEval",
    "completion",
    split="test",
    cache_dir="./data/mceval"
)
ds_mceval.to_json("./data/mceval/mceval-completion.jsonl")
print("McEval done:", len(ds_mceval), "rows")
print("columns:", ds_mceval.column_names)
print(ds_mceval[0])

with open("data/mceval/mceval-completion.jsonl", "rb") as f_in, \
     gzip.open("data/mceval/mceval-completion.jsonl.gz", "wb") as f_out:
    shutil.copyfileobj(f_in, f_out)

# SAFIM
ds_safim = load_dataset(
    "gonglinyuan/safim",
    "block",
    split="test",
    cache_dir="./data/safim"
)
ds_safim.to_json("./data/safim/safim.jsonl")
print("SAFIM done:", len(ds_safim), "rows")
print("columns:", ds_safim.column_names)
print(ds_safim[0])

# | Benchmark | Languages | Total Samples | Task Types | Data Source | Metric |
# |---|---|---|---|---|---|
# | HumanEval Infilling | Python only |  5815 | multi-line (pick this subset onlly) | HumanEval adapted | pass@k |
# | SAFIM (ICML 2024) | Multiple (verify exact list) | 17,720 | algorithmic block, control-flow expression, API function call | code commits (post-Apr 2022) | pass@k |
# | McEval-Completion | 40 languages | 12,000 | single-line, multi-line, span, span-light | manually annotated | pass@k |