#!/bin/bash -l
#----------------------------------------------------------------------------
# run_1node_2gpu.sbatch — DDP con torchrun en 1 nodo x 2 A100 (punto
# intermedio de escalado entre BASELINE y 2x2).
#
# Uso (desde DISTRIBUTED/):  sbatch run_1node_2gpu.sbatch [args de train_ddp.py]
#----------------------------------------------------------------------------
#SBATCH -J bert-ddp-1x2
#SBATCH -o %x_%j.out
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:2
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH -t 01:30:00

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

cd "$SLURM_SUBMIT_DIR"
srun torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    train_ddp.py --tag ddp_1x2 "$@"
