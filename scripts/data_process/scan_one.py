"""Scan one dataset's preprocessor for length consistency.
Usage: scan_one.py <dataset_path> <fn_name>
"""
import sys
import datasets
from src.data.titan_tokenizer import LLaMA32Tokenizer
from src.data.titan_preprocessor import SumAttentionPreprocessor

path, fnname = sys.argv[1], sys.argv[2]

tok = LLaMA32Tokenizer("data/titan_tokenizer/original/tokenizer.model")
pre = SumAttentionPreprocessor(
    tokenizer=tok, max_len=4096, special_token_start=128011,
    mem_start=128254, mem_end=128255, reencode_num=0, max_memory_num=40,
)
ds = datasets.load_from_disk(path)["train"]
fn = getattr(pre, fnname)
n = len(ds)
print(f"[{fnname}] n={n}", flush=True)

exc = mism = 0
first_exc = None
mism_examples = []
report_every = max(1, n // 20)

for i in range(n):
    try:
        out = fn(ds[i])
    except Exception as e:
        exc += 1
        if first_exc is None:
            first_exc = (i, type(e).__name__, str(e))
        continue
    li = len(out["input_ids"])
    ll = len(out["labels"])
    ls = len(out.get("segment_ids", out["input_ids"]))
    if not (li == ll == ls):
        mism += 1
        if len(mism_examples) < 5:
            mism_examples.append((i, li, ll, ls))
    if (i + 1) % report_every == 0:
        print(f"[{fnname}] progress {i+1}/{n}  exc={exc}  mism={mism}", flush=True)

print(f"[{fnname}] FINAL n={n} exc={exc} mismatch={mism}", flush=True)
if first_exc:
    print(f"[{fnname}] first_exc idx={first_exc[0]} {first_exc[1]}: {first_exc[2][:200]}", flush=True)
for ex in mism_examples:
    print(f"[{fnname}] MISMATCH idx={ex[0]} input_ids={ex[1]} labels={ex[2]} segment_ids={ex[3]}", flush=True)
print(f"[{fnname}] SCAN DONE", flush=True)
