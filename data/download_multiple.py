# pip install datasets
import os
import gzip, shutil
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset

# McEval
# ds_mceval = load_dataset(
#     "Multilingual-Multimodal-NLP/McEval",
#     "completion",
#     split="test",
#     cache_dir="./data/mceval"
# )
# ds_mceval.to_json("./data/mceval/mceval-completion.jsonl")
# print("McEval done:", len(ds_mceval), "rows")
# print("columns:", ds_mceval.column_names)
# print(ds_mceval[0])
#
# with open("data/mceval/mceval-completion.jsonl", "rb") as f_in, \
#      gzip.open("data/mceval/mceval-completion.jsonl.gz", "wb") as f_out:
#     shutil.copyfileobj(f_in, f_out)

# SAFIM
from datasets import load_dataset, concatenate_datasets

configs = ["block", "control", "api"]
splits = [
    load_dataset("gonglinyuan/safim", cfg, split="test", cache_dir="./data/safim")
    for cfg in configs
]

ds_safim = concatenate_datasets(splits)
ds_safim.to_json("./data/safim/safim.jsonl")
print("SAFIM done:", len(ds_safim), "rows")
print("columns:", ds_safim.column_names)
print(ds_safim[0])

# Function-Level FIM task
# | Benchmark | Languages | Total Samples | Task Types | Data Source | Metric |
# |---|---|---|---|---|---|
# | HumanEval Infilling (OpenAI) | Python only |  5815 | multi-line (pick this subset onlly) | HumanEval adapted | pass@k |
# | SAFIM (ICML 2024) | Multiple (verify exact list) | 17,720 | algorithmic block, control-flow expression, API function call | code commits (post-Apr 2022) | pass@k |
# | McEval-Completion | 40 languages | 12,000 | single-line, multi-line, span, span-light | manually annotated | pass@k |

import json
from pathlib import Path

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

# HumanEval — load from existing file
humaneval = load_jsonl("data/HumanEval-MultiLineInfilling.jsonl")
print(f"HumanEval multi-line: {len(humaneval)} (all Python)")

# SAFIM
safim = load_jsonl("data/safim/safim.jsonl")
safim_py = [x for x in safim if x.get("lang", "").lower() == "python"]
print(f"SAFIM total: {len(safim)} | Python: {len(safim_py)}")

# McEval
mceval = load_jsonl("data/mceval/mceval-completion.jsonl")
mceval_py = [x for x in mceval if x.get("task_id", "").split("/")[0].lower() == "python"]
print(f"McEval total: {len(mceval)} | Python: {len(mceval_py)}")