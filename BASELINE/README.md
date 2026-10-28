# BASELINE — BERT-base en SQuAD, 1 GPU A100

Fine-tuning de `google-bert/bert-base-uncased` sobre SQuAD v1.1 (QA
extractiva) con un bucle de entrenamiento en PyTorch puro, en 1 GPU A100 de
FinisTerrae III. Es la referencia de tiempos para el entregable DISTRIBUTED.

## Ficheros clave

| Fichero | Qué hace |
|---|---|
| `train_baseline.py` | Entrenamiento (bucle explícito, bf16 autocast, AdamW fused, scheduler lineal con warmup). Mide tiempo por época y throughput, y guarda un JSON con el resultado. |
| `common.py` | Rutas (`$LUSTRE/hpctools`), tokenización de SQuAD (ventanas 384/stride 128) y construcción de datasets. |
| `prepare_data.py` | Descarga modelo y dataset a la caché de `$LUSTRE` (una vez, en el login). |
| `run_baseline.sbatch` | Job SLURM: 1 nodo, 1 A100, 32 CPUs. |

## Cómo lanzarlo

```bash
cd BASELINE
source $STORE/mypython/bin/activate && pip install -r ../requirements.txt   # una vez
python prepare_data.py                      # una vez, en el login
sbatch run_baseline.sbatch                  # entrenamiento completo
```

Prueba rápida: `sbatch run_baseline.sbatch --max-train-samples 2000 --epochs 1`.

## Configuración

Batch 32, lr 3e-5, 2 épocas, longitud 384, bf16, semilla 42. El tiempo
reportado es **solo entrenamiento** (sin tokenización ni carga de datos), con
`torch.cuda.synchronize()` al inicio y fin de cada época. Se incluye la
validación solo como *loss* (no EM/F1; el postprocesado de respuestas queda
fuera porque el objetivo es medir velocidad de entrenamiento).

## Tiempos

| Config | Épocas | Tiempo total (s) | Throughput (samples/s) | Mem. pico (GB) | Job |
|---|---|---|---|---|---|
| 1 × A100 | 2 | _pendiente_ | _pendiente_ | _pendiente_ | _pendiente_ |

_Los valores salen del JSON en `$LUSTRE/hpctools/runs/baseline_<jobid>.json`
(línea `RESULT` del log del job). Pendiente de ejecutar en FT3._

## Notas

- Los nodos de cómputo se ejecutan con `HF_HUB_OFFLINE=1`; por eso hace falta
  `prepare_data.py` antes del primer job.
- Los logs `slurm-*.out` / `*.out` están en `.gitignore`.
