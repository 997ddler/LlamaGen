# !/bin/bash
set -x

CUDA_VISIBLE_DEVICES=0,1 torchrun \
--nnodes=1 --nproc_per_node=2 --node_rank=0 \
--master_addr=127.0.0.1 --master_port=29501 \
tokenizer/tokenizer_image/vq_train.py "$@"