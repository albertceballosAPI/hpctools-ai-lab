# HPC Tools — Lab AI: acelerar BERT-base en SQuAD

Máster en HPC (UDC), asignatura HPC Tools. Fine-tuning de
`google-bert/bert-base-uncased` sobre SQuAD v1.1 (QA extractiva) en PyTorch,
ejecutado en FinisTerrae III (CESGA) con SLURM y GPUs A100.

## Entregables

| Entregable | Carpeta | Tag | Contenido |
|---|---|---|---|
| 1 — BASELINE | [`BASELINE/`](BASELINE/) | `BASELINE` | Entrenamiento en 1 GPU A100 |
| 2 — DISTRIBUTED | [`DISTRIBUTED/`](DISTRIBUTED/) | `DISTRIBUTED` | DDP con `torchrun`, 1×2 y 2×2 A100 |

Cada carpeta es autocontenida (scripts Python, bash y SLURM, y su `README.md`
con ficheros clave y tiempos). `main` solo recibe cada entregable cuando está
terminado; el tag marca el commit final de cada uno.

## Cómo reproducir (FT3)

Una vez, en el nodo de login (necesita internet):

```bash
# venv del curso (o cualquier venv con torch + transformers + datasets)
source $STORE/mypython/bin/activate
pip install -r requirements.txt
python BASELINE/prepare_data.py     # modelo y SQuAD en $LUSTRE/hpctools/hf
```

Después, todo por SLURM, nunca en el login:

```bash
sbatch BASELINE/run_baseline.sbatch            # 1 GPU
sbatch DISTRIBUTED/run_1node_2gpu.sbatch       # 1 nodo, 2 GPUs
sbatch DISTRIBUTED/run_ddp.sbatch              # 2 nodos x 2 GPUs
```

Datos, caché de HuggingFace y resultados van a `$LUSTRE/hpctools`, no al
repo ni a `$HOME` (ver `.gitignore`).
