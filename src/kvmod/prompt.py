"""Shared eval-prompt assembly for the nq/hqa/wiki/musique scripts.

Every ``*_eval.py`` built input_ids + segment_ids with a byte-identical block;
the only per-benchmark difference is *which* docs/titles get extracted (gold
fields). That common assembly now lives here.

A ``prompt_preset`` swaps ONLY the system prompt; the chat wrapper, the
per-doc ``Document [N](Title: T) text`` rendering, the KVLink memory
sentinels (mem_start/mem_end), the link-token (reencode) handling and the
bare-question placement are identical across presets:

  * ``kvlink`` (default) -- the original KVLink system text. Reproduces the
    pre-refactor behaviour byte-for-byte (existing result stems unchanged).
  * ``short``  -- the modularization_ablation short-answer system text
    ("Give a short, direct answer and do not output any other words."),
    which suppresses verbose generations and lifts EM/F1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

PROMPT_PRESETS = ("kvlink", "short")

# System prompt body per preset. Only this string changes between presets.
SYSTEM_PROMPTS = {
    # Original KVLink hqa/wiki/nq/musique eval wording.
    "kvlink": (
        "You are a intelligent AI assistant. Please answer questions based "
        "on the user's instruction. Below are some reference documents that "
        "may help you in answering the user's question."
    ),
    # modularization_ablation DEFAULT_SYSTEM (short-answer / high-EM preset).
    "short": (
        "You are a helpful assistant. Answer the question based on the "
        "provided passages. Give a short, direct answer and do not output "
        "any other words."
    ),
}

# Llama-3 chat wrapper around the system text; the user turn opens right
# after, then [mem_start] docs [mem_end] question (all built below).
_CHAT_WRAPPER = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
)


@dataclass(frozen=True)
class ChatFormat:
    """Model-family chat framing. ``opening`` is everything up to (and incl.)
    the opener of the user turn, with a ``{system}`` slot; docs/question are
    appended after it. ``gen_prompt`` closes the user turn and opens the
    assistant turn (fed at generation time). ``stop_strings`` end a turn.
    ``assistant_split`` / ``stop_split`` extract the answer from the decoded
    text (response = text.split(assistant_split)[-1].split(stop_split)[0])."""
    name: str
    opening: str
    gen_prompt: str
    stop_strings: Tuple[str, ...]
    assistant_split: str
    stop_split: str


LLAMA_FORMAT = ChatFormat(
    name="llama",
    opening=_CHAT_WRAPPER,
    gen_prompt="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    stop_strings=("<|eot_id|>", "<|end_of_text|>"),
    assistant_split="<|start_header_id|>assistant<|end_header_id|>",
    stop_split="<|eot_id|>",
)

# Qwen2/Qwen2.5 ChatML framing (no BOS token; <|im_start|>/<|im_end|>).
QWEN_FORMAT = ChatFormat(
    name="qwen",
    opening=("<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n"),
    gen_prompt="<|im_end|>\n<|im_start|>assistant\n",
    stop_strings=("<|im_end|>", "<|endoftext|>"),
    assistant_split="<|im_start|>assistant\n",
    stop_split="<|im_end|>",
)

# Qwen3 ChatML in NON-THINKING mode: the official template suppresses reasoning
# by injecting an empty ``<think>\n\n</think>`` block right after the assistant
# tag, so the model answers directly. The answer is everything after the
# (empty) think block, so split on ``</think>``.
QWEN3_FORMAT = ChatFormat(
    name="qwen3",
    opening=("<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n"),
    gen_prompt="<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    stop_strings=("<|im_end|>", "<|endoftext|>"),
    assistant_split="</think>",
    stop_split="<|im_end|>",
)

CHAT_FORMATS = {"llama": LLAMA_FORMAT, "qwen": QWEN_FORMAT, "qwen3": QWEN3_FORMAT}


def get_chat_format(model_name: str) -> ChatFormat:
    """Pick the chat framing by model family (defaults to Llama)."""
    n = model_name.lower()
    if "qwen3" in n:
        return QWEN3_FORMAT
    if "qwen" in n:
        return QWEN_FORMAT
    return LLAMA_FORMAT


def system_text(preset: str, chat_format: ChatFormat = LLAMA_FORMAT) -> str:
    if preset not in SYSTEM_PROMPTS:
        raise ValueError(
            f"Unknown prompt_preset {preset!r}; expected one of {PROMPT_PRESETS}"
        )
    return chat_format.opening.format(system=SYSTEM_PROMPTS[preset])


def build_segmented_inputs(
    tokenizer,
    question: str,
    docs: Sequence[Tuple[str, str]],
    *,
    prompt_preset: str,
    reencode_num: int,
    mem_start: int,
    mem_end: int,
    special_token_start: int,
    chat_format: ChatFormat = LLAMA_FORMAT,
) -> Dict[str, List[int]]:
    """Assemble input_ids + segment_ids (the shared body of preprocess_fn).

    ``docs`` is an ordered list of ``(title, text)``; the caller does the
    per-benchmark gold/distractor extraction. ``segment_ids``: 0 = system
    prefix / link tokens / question, k>=1 = document k (1-indexed). This is
    byte-identical to the old hardcoded assembly for ``prompt_preset
    == "kvlink"``.
    """
    if chat_format is None:
        chat_format = LLAMA_FORMAT
    memory_list = [
        f"Document [{j + 1}](Title: {title}) {text}\n"
        for j, (title, text) in enumerate(docs)
    ]

    system = system_text(prompt_preset, chat_format)
    qa_system_input_ids = tokenizer(system, add_special_tokens=False)["input_ids"]
    input_ids = qa_system_input_ids[:] + [mem_start]
    segment_ids = [0] * len(input_ids)

    for mem_id, st in enumerate(memory_list):
        tem_id = tokenizer(st, add_special_tokens=False)["input_ids"]
        segment_ids = segment_ids + [mem_id + 1] * len(tem_id) + [0] * reencode_num

        for sub_idx in range(reencode_num):
            tem_id = tem_id + [special_token_start + reencode_num * mem_id + sub_idx]
        input_ids = input_ids + tem_id

    prompt_id = tokenizer(question, add_special_tokens=False)["input_ids"]
    input_ids = input_ids + [mem_end] + prompt_id
    segment_ids = segment_ids + [0] + [0] * len(prompt_id)
    return {"input_ids": input_ids, "segment_ids": segment_ids}
