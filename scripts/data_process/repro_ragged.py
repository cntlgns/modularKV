"""Reproduce + identify which preprocess fn produces the ragged sample.

Monkey-patches every preprocess_* on SumAttentionPreprocessor so that each
call records (fn_name, len(input_ids), len(labels), len(segment_ids), sample_keys)
and dumps the offender when a batch is ragged.
"""
import sys

from src.data.titan_data_utils import build_hf_data_loader
from src.data.titan_preprocessor import BlockAttnCollator, SumAttentionPreprocessor
from src.data.titan_tokenizer import LLaMA32Tokenizer
from src.training.titan_training_utils import DATASET_MAPPING

MAX_BATCHES = int(sys.argv[1]) if len(sys.argv) > 1 else 200
BATCH_SIZE = 8

tok = LLaMA32Tokenizer("data/titan_tokenizer/original/tokenizer.model")
pre = SumAttentionPreprocessor(
    tokenizer=tok, max_len=4096, special_token_start=128011,
    mem_start=128254, mem_end=128255, reencode_num=0, max_memory_num=40,
)

# Monkey-patch each process_* on pre to record its name + lengths on the output.
import functools
for name in list(dir(pre)):
    if not name.startswith("process_"):
        continue
    orig = getattr(pre, name)
    if not callable(orig):
        continue
    @functools.wraps(orig)
    def wrapped(example, __orig=orig, __name=name):
        out = __orig(example)
        out["_fn"] = __name
        out["_lens"] = (len(out.get("input_ids", [])),
                        len(out.get("labels", [])),
                        len(out.get("segment_ids", [])))
        return out
    setattr(pre, name, wrapped)

inner = BlockAttnCollator(pad_token_idx=tok.pad_id)

class DiagCollator:
    def __call__(self, features):
        bad = []
        for idx, f in enumerate(features):
            li, ll, ls = f.get("_lens", (len(f.get("input_ids", [])),
                                         len(f.get("labels", [])),
                                         len(f.get("segment_ids", []))))
            if not (li == ll == ls):
                bad.append((idx, li, ll, ls, f.get("_fn", "?")))
        if bad:
            print(f"\n*** RAGGED BATCH: {len(bad)} mismatched features ***", flush=True)
            for idx, li, ll, ls, fn in bad:
                print(f"  feature[{idx}] fn={fn} input_ids={li} labels={ll} segment_ids={ls}", flush=True)
            # show other features' source too for context
            print("  all features in this batch:", flush=True)
            for idx, f in enumerate(features):
                li, ll, ls = f.get("_lens", (0, 0, 0))
                print(f"    [{idx}] fn={f.get('_fn','?')} ({li},{ll},{ls})", flush=True)
            raise SystemExit(0)
        # strip our metadata before forwarding to real collator
        clean = []
        for f in features:
            c = {k: v for k, v in f.items() if not k.startswith("_")}
            clean.append(c)
        return inner(clean)

loader = build_hf_data_loader(
    DATASET_MAPPING["original"],
    tokenizer=tok,
    preprocessor=pre,
    seed=42,
    batch_size=BATCH_SIZE,
    seq_len=4096,
    world_size=1, rank=0,
    collate_fn=DiagCollator(),
    infinite=True,
    enable_packing=False,
)

it = iter(loader)
for b in range(MAX_BATCHES):
    batch = next(it)
    if b % 5 == 0:
        print(f"batch {b}: ok  shapes input_ids={tuple(batch['input_ids'].shape)}", flush=True)
print(f"NO RAGGED in {MAX_BATCHES} batches", flush=True)
