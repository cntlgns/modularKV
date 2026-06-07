#!/bin/bash
# All three attention analyses on the 8B base model, one GPU.
set -e
MODEL="${1:-meta-llama/Llama-3.1-8B-Instruct}"
cd /data_fast/home/sihun/kvcache/KVLink
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')  MODEL=$MODEL"

echo "######## 1/3 doc-block region split ########"
python scripts/evaluation/attn_doc_block.py --model "$MODEL" \
    --datasets nq block_qa hqa_full wiki musique --policies baseline recover_pos_enc \
    --n_samples 10 --out_dir result/attn_doc_block --device cuda --dtype bfloat16

echo "######## 2/3 within-doc (layer x head) ########"
python scripts/evaluation/attn_within_doc.py --model "$MODEL" \
    --datasets nq block_qa hqa_full wiki musique --policies baseline recover_pos_enc \
    --n_samples 10 --out_dir result/attn_within_doc --device cuda --dtype bfloat16

echo "######## 3/3 q.k^T decomposition ########"
python scripts/evaluation/attn_qk_decompose.py --model "$MODEL" \
    --datasets nq musique --policies baseline recover_pos_enc \
    --n_samples 8 --out_dir result/attn_qk_decompose --device cuda --dtype bfloat16
