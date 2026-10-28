#!/bin/bash -l
#----------------------------------------------------------------------------
# run_baseline.sbatch — BASELINE: BERT-base en SQuAD con 1 GPU A100 en FT3.
#
# Uso (desde BASELINE/):
#   sbatch run_baseline.sbatch                      # entrenamiento completo
#   sbatch run_baseline.sbatch --max-train-samples 2000 --epochs 1   # prueba
# Antes, UNA vez en el login:  python prepare_data.py
# Otro entorno:  VENV=$STORE/otro sbatch run_baseline.sbatch
#----------------------------------------------------------------------------
#SBATCH -J bert-baseline
#SBATCH -o %x_%j.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH -c 32
#SBATCH --mem=32G
#SBATCH -t 02:00:00

set -euo pipefail

VENV="${VENV:-$STORE/mypython}"
if [ ! -f "$VENV/bin/activate" ]; then
    echo "No existe el entorno $VENV" >&2
    exit 1
fi
source "$VENV/bin/activate"

# Datos y caché en $LUSTRE (nodos de cómputo sin internet: modo offline)
export HF_HOME="$LUSTRE/hpctools/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONFAULTHANDLER=1

cd "$SLURM_SUBMIT_DIR"
python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
srun python train_baseline.py "$@"
