"""
Batch-API answer generation for block_qa.

Reads the existing block_qa.jsonl checkpoint and fills in every entry that
still lacks a `generated` answer via the OpenAI Batch API (50% cheaper than
sync). Tier-1 accounts cap enqueued batch tokens at 2M, so requests are sent
in sequential token-budgeted chunks: submit one batch, wait for it, write the
answers back, then submit the next.

    python scripts/data_process/gpt_answer_batch.py

Resumable: block_qa.jsonl is rewritten after every chunk, and an in-flight
batch id is persisted to batch/inflight.json, so a re-run after an
interruption recovers the pending batch and continues.
"""
import json
import os
import time

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("api_key"))

OUT_PATH = "data/raw/block_qa/block_qa.jsonl"
BATCH_DIR = "data/raw/block_qa/batch"
INFLIGHT_PATH = os.path.join(BATCH_DIR, "inflight.json")
SYSTEM_MSG = (
    "You are an intelligent AI assistant. Please answer questions based on "
    "the user's instructions. Below are some reference documents that may "
    "help you in answering the user's question.\n\n"
)
MAX_TOKENS = 200
# Tier-1 enqueued-token cap is 2M; stay well under it per batch.
TOKEN_BUDGET = 1_700_000

_enc = tiktoken.encoding_for_model("gpt-4o-mini")
_SYS_TOK = len(_enc.encode(SYSTEM_MSG))


def build_user_msg(d):
    user_msg = ""
    for j, doc in enumerate(d["documents"]):
        user_msg += f"Document [{j+1}](Title: {doc['title']}) {doc['text']}\n"
    user_msg += d["question"]
    return user_msg


def enqueued_tokens(user_msg):
    return _SYS_TOK + len(_enc.encode(user_msg)) + MAX_TOKENS


def make_request(idx, user_msg):
    return {
        "custom_id": str(idx),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.7,
            "max_tokens": MAX_TOKENS,
        },
    }


def poll(bid):
    while True:
        b = client.batches.retrieve(bid)
        rc = b.request_counts
        print(f"  {bid}: {b.status} ({rc.completed}/{rc.total}, failed={rc.failed})", flush=True)
        if b.status in ("completed", "failed", "expired", "cancelled"):
            return b
        time.sleep(60)


def collect(batch, data):
    if batch.status == "failed":
        errs = getattr(batch, "errors", None)
        raise RuntimeError(f"batch {batch.id} failed: {errs}")
    filled = 0
    if batch.output_file_id:
        content = client.files.content(batch.output_file_id).text
        for line in content.splitlines():
            r = json.loads(line)
            idx = int(r["custom_id"])
            resp = r.get("response")
            if resp and resp.get("status_code") == 200:
                data[idx]["generated"] = resp["body"]["choices"][0]["message"]["content"]
                filled += 1
    return filled


def save(data):
    tmp = OUT_PATH + ".tmp"
    with open(tmp, "w") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    os.replace(tmp, OUT_PATH)


def run_chunk(data, indices, chunk_no):
    path = os.path.join(BATCH_DIR, f"input_{chunk_no}.jsonl")
    with open(path, "w") as f:
        for i in indices:
            f.write(json.dumps(make_request(i, build_user_msg(data[i]))) + "\n")
    up = client.files.create(file=open(path, "rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=up.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"chunk {chunk_no}: submitted {batch.id} ({len(indices)} requests)", flush=True)
    with open(INFLIGHT_PATH, "w") as f:
        json.dump({"batch_id": batch.id, "chunk_no": chunk_no}, f)

    batch = poll(batch.id)
    filled = collect(batch, data)
    save(data)
    os.remove(INFLIGHT_PATH)
    print(f"chunk {chunk_no}: filled {filled} answers", flush=True)


def main():
    os.makedirs(BATCH_DIR, exist_ok=True)
    with open(OUT_PATH) as f:
        data = [json.loads(line) for line in f]

    # Recover an in-flight batch from a previous interrupted run.
    if os.path.exists(INFLIGHT_PATH):
        with open(INFLIGHT_PATH) as f:
            inflight = json.load(f)
        print(f"recovering in-flight batch {inflight['batch_id']}", flush=True)
        batch = poll(inflight["batch_id"])
        collect(batch, data)
        save(data)
        os.remove(INFLIGHT_PATH)

    chunk_no = 0
    while True:
        todo = [i for i, d in enumerate(data) if not d.get("generated")]
        done = len(data) - len(todo)
        print(f"total={len(data)} done={done} todo={len(todo)}", flush=True)
        if not todo:
            break

        # Take the next token-budgeted chunk.
        chunk, budget = [], 0
        for i in todo:
            t = enqueued_tokens(build_user_msg(data[i]))
            if chunk and budget + t > TOKEN_BUDGET:
                break
            chunk.append(i)
            budget += t
        print(f"chunk {chunk_no}: {len(chunk)} requests, ~{budget:,} enqueued tokens", flush=True)
        run_chunk(data, chunk, chunk_no)
        chunk_no += 1

    print(f"all done: {sum(1 for d in data if d.get('generated'))}/{len(data)}", flush=True)


if __name__ == "__main__":
    main()
