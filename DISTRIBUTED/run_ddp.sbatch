#!/bin/bash -l
#----------------------------------------------------------------------------
# run_ddp.sbatch — DDP con torchrun en 2 nodos x 2 A100 (4 GPUs) en FT3.
#
# Uso (desde DISTRIBUTED/):
#   sbatch run_ddp.sbatch                                   # completo
#   sbatch run_ddp.sbatch --max-train-samples 2000 --epochs 1   # prueba
# Antes, UNA vez en el login:  python prepare_data.py
#
# Convención: UNA tarea SLURM por nodo; `srun torchrun` en cada nodo lanza
# 2 procesos (uno por GPU). El rendezvous c10d usa el primer nodo como host.
# (Distinto a sesion7_lightning, que lanza una tarea SLURM por GPU sin torchrun.)
#----------------------------------------------------------------------------
#SBATCH -J bert-ddp-2x2
#SBATCH -o %x_%j.out
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:2
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH -t 01:00:00

set -euo pipefail

VENV="${VENV:-$STORE/mypython}"
if [ ! -f "$VENV/bin/activate" ]; then
    echo "No existe el entorno $VENV" >&2
    exit 1
fi
source "$VENV/bin/activate"

export HF_HOME="$LUSTRE/hpctools/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONFAULTHANDLER=1
# export NCCL_DEBUG=INFO     # depuración de comunicaciones (opcional)

GPUS_PER_NODE=2
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
RDZV_PORT=29500

cd "$SLURM_SUBMIT_DIR"
echo "Nodos: $SLURM_JOB_NODELIST | head: $HEAD_NODE:$RDZV_PORT | GPUs/nodo: $GPUS_PER_NODE"

srun torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node="$GPUS_PER_NODE" \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$HEAD_NODE:$RDZV_PORT" \
    train_ddp.py --tag ddp_2x2 "$@"
