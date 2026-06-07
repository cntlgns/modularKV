"""
Evaluate the performance of LLMs fine-tuned via TorchTitan on wiki dataset.

1. Convert the Titan checkpoint from DCP to torch:
```
python -m torch.distributed.checkpoint.format_utils dcp_to_torch \
    torchtitan/outputs/checkpoint/step-1000 checkpoint.pt
```

2. Run evaluation:
```
python scripts/evaluation/wiki_eval.py \
    --ckpt_path checkpoint.pt \
    --batch_size 10 \
    --reencode_num 5 \
    --attn_type "blocked" \
```
"""
import argparse
import datetime
import json
import os
from typing import Dict, List

import datasets
import numpy as np
import torch
from safetensors import safe_open
from torch.utils.data import DataLoader
from torchtune.models.convert_weights import tune_to_hf
from tqdm.auto import tqdm as auto_tqdm
from transformers import AutoTokenizer, GenerationConfig, LlamaForCausalLM, AutoModelForCausalLM

from src.common import move_to_target_device
from src.data.titan_preprocessor import LLaMA32Tokenizer
from src.kvmod import (
    KV_CACHE_POLICIES,
    MODULAR_Q_POS_CHOICES,
    POLICIES_WITH_MODULAR_Q_POS,
    PROMPT_PRESETS,
    build_segmented_inputs,
    generate_for_policy,
    get_chat_format,
)
from src.utils.metrics import score_sample

# Legacy --attn_type aliases (pre-5-policy refactor).
_POLICY_ALIASES = {"standard": "baseline", "blocked": "recover_pos_enc"}

parser = argparse.ArgumentParser(description="Run script with specified ckpt.")
parser.add_argument(
    "--ckpt_path",
    type=str,
    default=None,
    help="The path to the `checkpoint.pt` file.",
)
parser.add_argument("--batch_size", type=int, default=1, help="Batch size of the evaluation.")
parser.add_argument(
    "--kv_policy",
    type=str,
    required=True,
    help="KV-cache modularization policy (5 names shared with the "
    "modularization_ablation repo). 'standard'/'blocked' are accepted as "
    "legacy aliases for 'baseline'/'recover_pos_enc'.",
    choices=list(KV_CACHE_POLICIES) + list(_POLICY_ALIASES),
)
parser.add_argument(
    "--modular_q_pos",
    type=str,
    default="summed_pos",
    choices=list(MODULAR_Q_POS_CHOICES),
    help="Question RoPE placement for reset-to-zero policies "
    "(modular / recover_cross_attn); ignored otherwise.",
)
parser.add_argument(
    "--prompt_preset",
    type=str,
    default="kvlink",
    choices=list(PROMPT_PRESETS),
    help="System-prompt preset. 'kvlink' = original wording (default, "
    "byte-identical to pre-refactor). 'short' = short-answer system prompt "
    "(higher EM/F1). Doc rendering / framing unchanged either way.",
)
parser.add_argument("--reencode_num", type=int, default=5, help="Number of the link tokens.")
parser.add_argument("--hf", type=bool, default=False, help="Use HuggingFace checkpoints.")
parser.add_argument(
    "--model_name",
    type=str,
    default="",
    help="HF model/tokenizer id. Defaults to ckpt_path when --hf, else Llama-3.2-1B-Instruct.",
)
parser.add_argument(
    "--gold_only",
    action="store_true",
    help="Feed only the gold (supporting_facts) documents instead of all distractor documents.",
)
parser.add_argument(
    "--max_examples",
    type=int,
    default=None,
    help="If set, evaluate on at most this many examples (clamped to len(dataset)).",
)
parser.add_argument(
    "--out_dir",
    type=str,
    default="result/ablation",
    help="Output directory for the .jsonl + .summary.json (relative to project root).",
)

args = parser.parse_args()

def load_model_weights(ckpt_path: str):
    safe_tensor_file = os.path.join(ckpt_path, "model.safetensors")
    if os.path.exists(safe_tensor_file):
        state_dict = {}
        with safe_open(safe_tensor_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        # state_dict["output.weight"] = state_dict["tok_embeddings.weight"]
        return state_dict

    state_dict = torch.load(ckpt_path, weights_only=False)

    state_dict = state_dict["model"]
    state_dict["output.weight"] = state_dict["tok_embeddings.weight"]

    converted_state_dict = tune_to_hf(
        state_dict=state_dict,
        num_heads=32,
        num_kv_heads=8,
        dim=2048,
    )
    return converted_state_dict

# QA metrics (normalize_answer / best_subspan_em / exact_match / f1_score)
# were consolidated into src.utils.metrics -- see score_sample, imported above.
# substring accuracy is byte-for-byte unchanged from the old inline version.

def preprocess_fn(example: Dict[str, str], tokenizer: LLaMA32Tokenizer, reencode_num: int, mem_start: int, mem_end: int, special_token_start:int, prompt_preset: str = "kvlink", gold_only: bool = False, chat_format=None):
    question = example["question"]
    gold_titles = {sf['title'] for sf in example['supporting_facts']} if gold_only else None
    docs = []
    for j in range(0, 10):
        title = example['context'][j]['title']
        if gold_only and title not in gold_titles:
            continue
        text = " ".join(example['context'][j]['sentences'])
        docs.append((title, text))

    return build_segmented_inputs(
        tokenizer, question, docs,
        prompt_preset=prompt_preset, reencode_num=reencode_num,
        mem_start=mem_start, mem_end=mem_end,
        special_token_start=special_token_start,
        chat_format=chat_format,
    )

class DataCollatorForGeneration():
    def __init__(self, pad_id: int):
        self.pad_id = pad_id
        pass
    def __call__(self, batch):
        input_ids = []
        segment_ids = []
        attention_mask = []
        length_list = [len(x['input_ids']) for x in batch]

        max_length = max(length_list)

        for item in batch:
            seq_length = len(item['input_ids'])

            residual = max_length - seq_length
            # padded_input_ids = [self.pad_id] * residual + item['input_ids']
            # curr_attention_mask = [0] * residual + [1] * seq_length
            padded_input_ids = item['input_ids'] + [self.pad_id] * residual
            curr_attention_mask = [1] * seq_length + [0] * residual
            input_ids.append(padded_input_ids)
            attention_mask.append(curr_attention_mask)
            segment_ids.append(item["segment_ids"] + [-1] * residual)

        return {
            "input_ids": torch.LongTensor(input_ids),
            "segment_ids": torch.LongTensor(segment_ids),
            "attention_mask": torch.LongTensor(attention_mask),
        }


def main():
    ckpt_path = args.ckpt_path
    reencode_num: int  = args.reencode_num
    batch_size: int = args.batch_size
    hf: bool = args.hf
    policy = _POLICY_ALIASES.get(args.kv_policy, args.kv_policy)

    device = torch.device("cuda")

    mem_start = 128254
    mem_end = 128255
    special_token_start = 128011

    data_path="data/raw/2wikimultihop/dev.json"
    dataset = datasets.load_dataset("json", data_files=data_path, split="train")
    print(dataset)
    all_answers = dataset["answer"]
    print(all_answers[:10])

    model_name = args.model_name or (args.ckpt_path if (hf and args.ckpt_path) else "meta-llama/Llama-3.2-1B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    chat_fmt = get_chat_format(model_name)
    if "Llama-3.2" in model_name:
        # Original KVLink setup: reserved Llama-3.2 special-token ids.
        tokenizer.pad_token_id = 128004
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
    else:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Re-map segment sentinels into this model's own vocab range.
        vocab_size = len(tokenizer)
        mem_start = vocab_size - 2
        mem_end = vocab_size - 1
        special_token_start = vocab_size - 2 - 1024

    if hf and args.ckpt_path is not None:
        model = AutoModelForCausalLM.from_pretrained(args.ckpt_path, torch_dtype=torch.bfloat16)
    elif args.ckpt_path is None:
        print("Will NOT load fine-tuned models! Evaluating pretrained model_name directly.")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    else:
        model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        state_dict = load_model_weights(ckpt_path)
        model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    exist_columns = dataset.column_names
    dataset = dataset.map(
        preprocess_fn,
        batched=False,
        num_proc=16,
        remove_columns=exist_columns,
        fn_kwargs=dict(
            tokenizer=tokenizer,
            reencode_num=reencode_num,
            mem_start=mem_start,
            mem_end=mem_end,
            special_token_start=special_token_start,
            prompt_preset=args.prompt_preset,
            chat_format=chat_fmt,
            gold_only=args.gold_only,
        ),
    )

    total_num = min(args.max_examples, len(dataset)) if args.max_examples else len(dataset)
    dataset = dataset.select(np.arange(total_num))
    all_answers = all_answers[:total_num]
    correct_num = 0
    em_sum = 0.0
    f1_sum = 0.0
    res_list = []

    collate_fn = DataCollatorForGeneration(pad_id=tokenizer.pad_token_id)
    eval_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    prog_bar = auto_tqdm(range(len(eval_dataloader)))

    # Greedy stop ids: original eval used GenerationConfig(eos=<|eot_id|>,
    # stop_strings=["<|end_of_text|>","<|eot_id|>"]). The kvmod greedy loop
    # stops on these ids (token-for-token identical to the old greedy decode).
    stop_token_ids = {
        i for i in (
            *[tokenizer.convert_tokens_to_ids(s) for s in chat_fmt.stop_strings],
            tokenizer.eos_token_id,
        ) if i is not None and i >= 0
    }
    generation_prompt = chat_fmt.gen_prompt
    generation_token_ids = tokenizer(generation_prompt, add_special_tokens=False)["input_ids"]
    generation_token_ids = torch.LongTensor(generation_token_ids)
    generation_token_ids = move_to_target_device(generation_token_ids, device)

    for batch_id, batch in enumerate(eval_dataloader):
        curr_batch_size = batch['input_ids'].size(0)
        batch_answers = all_answers[batch_id * batch_size : batch_id * batch_size + curr_batch_size]

        responses = generate_for_policy(
            model,
            tokenizer,
            input_ids=batch["input_ids"],
            pad_mask=batch["attention_mask"],
            segment_ids=batch["segment_ids"],
            policy=policy,
            generation_token_ids=generation_token_ids,
            stop_token_ids=stop_token_ids,
            assistant_split=chat_fmt.assistant_split,
            stop_split=chat_fmt.stop_split,
            max_new_tokens=200,
            modular_q_pos=args.modular_q_pos,
        )
        for idx, x in enumerate(responses):
            print(x)
            print("Ground-truth: ", batch_answers[idx])
            print("------\n")

        scored = [score_sample(responses[idx], [batch_answers[idx]]) for idx in range(curr_batch_size)]
        for idx, s in enumerate(scored):
            correct_num = correct_num + int(s["substring"])
            em_sum = em_sum + s["em"]
            f1_sum = f1_sum + s["f1"]
            res_list.append(
                {
                    # "question": question,
                    "response": responses[idx],
                    "gold_answer": batch_answers[idx],
                    "score": s["substring"],
                    "em": s["em"],
                    "f1": s["f1"],
                }
            )
        print("Correct progress", correct_num)
        prog_bar.update(1)

    accuracy = correct_num / total_num
    em = em_sum / total_num
    f1 = f1_sum / total_num
    print(f"substring-acc={accuracy}  em={em}  f1={f1}")

    current_time = datetime.datetime.now()
    time_str = current_time.strftime("%Y%m%d-%H%M%S")

    state = "goldonly" if args.gold_only else "all"
    model_slug = model_name.split("/")[-1]
    # Deterministic, ablation-unique stem: distinct experiments never overwrite
    # each other, and sweep_ablation.py skips a job whose .jsonl already exists.
    # accuracy/timestamp live in the sidecar .summary.json, not the filename.
    # Keep in sync with sweep_ablation.result_stem().
    qpos = f"-{args.modular_q_pos}" if policy in POLICIES_WITH_MODULAR_Q_POS else ""
    ppx = "" if args.prompt_preset == "kvlink" else f"_p-{args.prompt_preset}"
    stem = f"wiki_{model_slug}_{state}_kv-{policy}{qpos}{ppx}_re{reencode_num}"
    out_dir = args.out_dir
    file_name = f"{out_dir}/{stem}.jsonl"
    os.makedirs(out_dir, exist_ok=True)

    with open(file_name, "w", encoding="utf-8") as f:
        for entry in res_list:
            f.write(json.dumps(entry) + "\n")

    summary = {
        "stem": stem, "benchmark": "wiki", "model": model_name,
        "state": state, "kv_policy": policy, "modular_q_pos": args.modular_q_pos,
        "prompt_preset": args.prompt_preset,
        "reencode_num": reencode_num,
        "accuracy": accuracy, "em": em, "f1": f1,
        "num_examples": total_num, "timestamp": time_str,
    }
    with open(f"{out_dir}/{stem}.summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Dumped at {file_name}  (acc={accuracy} em={em} f1={f1})")

if __name__ == "__main__":
    main()

