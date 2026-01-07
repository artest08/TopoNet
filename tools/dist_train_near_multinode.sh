#!/usr/bin/env bash
set -x

timestamp=$(date +"%y%m%d.%H%M%S")

WORK_DIR=work_dirs/toponet_near_v2
CONFIG=projects/configs/toponet_r50_8x1_24e_olv2_subset_A_near.py

GPUS=${1:-${SLURM_GPUS_ON_NODE:-1}}
NNODES=${SLURM_NNODES:-1}
NODE_RANK=${SLURM_NODEID:-0}
PORT=${PORT:-28510}

if [[ -z "${MASTER_ADDR}" ]]; then
  if [[ -n "${SLURM_JOB_NODELIST}" ]] && command -v scontrol >/dev/null 2>&1; then
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
  else
    MASTER_ADDR=127.0.0.1
  fi
fi

~/containers/python_topomlp -m torch.distributed.run \
  --nproc_per_node=$GPUS \
  --nnodes=$NNODES \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$PORT \
  tools/train.py $CONFIG --launcher pytorch --work-dir ${WORK_DIR} --deterministic ${@:2} \
  2>&1 | tee ${WORK_DIR}/train.${timestamp}.rank${NODE_RANK}.log
