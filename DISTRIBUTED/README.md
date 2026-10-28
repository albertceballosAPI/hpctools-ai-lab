# DISTRIBUTED — BERT-base en SQuAD con DDP, 2 nodos × 2 A100

Versión distribuida del BASELINE con **DistributedDataParallel** (PyTorch
nativo) lanzada con `torchrun`. Probada en 1×2 y en 2×2 A100 de FinisTerrae III.

## Ficheros clave

| Fichero | Qué hace |
|---|---|
| `train_ddp.py` | Entrenamiento DDP: `init_process_group("nccl")`, `DistributedSampler` con `set_epoch`, logs y guardado solo en rank 0, `destroy_process_group`. |
| `run_ddp.sbatch` | 2 nodos × 2 A100: `srun torchrun --nnodes=2 --nproc_per_node=2` con rendezvous c10d en el primer nodo. |
| `run_1node_2gpu.sbatch` | 1 nodo × 2 A100 (punto intermedio de escalado). |
| `common.py`, `prepare_data.py` | Rutas, tokenización y descarga de datos (idénticos a BASELINE). |

## Qué cambia respecto al BASELINE

- Modelo envuelto en `DDP(model, device_ids=[local_rank])`; el all-reduce de
  gradientes se solapa con el backward.
- `DistributedSampler(drop_last=True)` reparte los datos entre ranks;
  `set_epoch(epoch)` cambia el barajado en cada época.
- El rank 0 tokeniza y escribe la caché de `datasets` en `$LUSTRE` (compartido
  entre nodos); el resto espera con un `barrier` y la lee.
- Métricas (loss, tiempos) agregadas entre ranks; solo el rank 0 imprime y
  escribe el JSON de resultados.
- Misma medición de tiempo que el BASELINE (solo entrenamiento, con
  `cuda.synchronize()` + `barrier` al inicio y fin de cada época).

## Cómo lanzarlo

```bash
cd DISTRIBUTED
python prepare_data.py                 # una vez, en el login (si no se hizo ya)
sbatch run_1node_2gpu.sbatch           # 1 nodo, 2 GPUs
sbatch run_ddp.sbatch                  # 2 nodos, 4 GPUs
```

Prueba rápida: `sbatch run_ddp.sbatch --max-train-samples 2000 --epochs 1`.

## Batch por GPU frente a batch global

`--batch-size` es **por GPU** (32 por defecto). Con DDP el batch global es
`32 × nº de GPUs`: 32 (BASELINE), 64 (1×2) y 128 (2×2). Se mantiene el mismo
`lr` (3e-5) y el mismo nº de épocas, así que hay **menos pasos de optimización**
por época al escalar. Esto mide el *weak scaling del batch* (más datos por
paso), no el mismo cómputo repartido; la loss final puede variar ligeramente.
Si se quiere el mismo batch global que el BASELINE: `--batch-size 16` (1×2) y
`--batch-size 8` (2×2), a costa de más comunicación por muestra.

## Tiempos

Speedup = T(1 GPU) / T(N GPUs); eficiencia = speedup / N.

| Config | GPUs | Batch/GPU | Batch global | Tiempo total (s) | Throughput (samples/s) | Speedup | Eficiencia | Job |
|---|---|---|---|---|---|---|---|---|
| BASELINE 1 × A100 | 1 | 32 | 32 | _pendiente_ | _pendiente_ | 1,00 | 100 % | _pendiente_ |
| DDP 1 nodo × 2 A100 | 2 | 32 | 64 | _pendiente_ | _pendiente_ | _pendiente_ | _pendiente_ | _pendiente_ |
| DDP 2 nodos × 2 A100 | 4 | 32 | 128 | _pendiente_ | _pendiente_ | _pendiente_ | _pendiente_ | _pendiente_ |

_Los valores salen de la línea `RESULT` del log (JSON en
`$LUSTRE/hpctools/runs/<tag>_<jobid>.json`). Pendiente de ejecutar en FT3.
Todas las medidas deben correr en las colas por defecto (mismo pool de nodos
Ice Lake / A100)._

## Profiling

_Pendiente (opcional): `torch.profiler` o `NCCL_DEBUG=INFO` para ver el
solape de comunicación y la diferencia intra-nodo (NVLink) frente a inter-nodo._

## Notas

- Nodos de cómputo en modo offline (`HF_HUB_OFFLINE=1`): hace falta
  `prepare_data.py` antes del primer job.
- `sesion7_lightning` lanza una tarea SLURM por GPU sin torchrun; aquí se usa
  una tarea por nodo y `torchrun` con rendezvous c10d, como pide el enunciado.
- Los logs `*.out` están en `.gitignore`; los que sirvan de evidencia de
  tiempos se añaden a propósito con `git add -f`.
