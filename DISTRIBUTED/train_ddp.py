"""DISTRIBUTED: fine-tuning de BERT-base en SQuAD v1.1 con DDP (PyTorch nativo).

Lanzar con torchrun (una GPU por proceso). Misma receta y misma medición de
tiempos que BASELINE/train_baseline.py: solo entrenamiento, con
cuda.synchronize() + barrier al inicio y fin de cada época.

--batch-size es POR GPU: el batch global es batch_size * world_size.
"""
import argparse
import json
import os
import time

import common

common.setup_hf_env()

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torch.utils.data.distributed import DistributedSampler  # noqa: E402
from transformers import (AutoModelForQuestionAnswering, AutoTokenizer,  # noqa: E402
                          get_linear_schedule_with_warmup)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32, help="batch por GPU")
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max-len", type=int, default=384)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--num-workers", type=int,
                   default=min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", 4))))
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-eval-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default="ddp")
    p.add_argument("--save-model", action="store_true",
                   help="guarda el modelo en $LUSTRE (solo rank 0)")
    return p.parse_args()


@torch.no_grad()
def eval_loss(model, loader, device, use_bf16):
    """Loss de validación promediada sobre todos los ranks."""
    model.eval()
    stats = torch.zeros(2, device=device)  # [suma de loss, nº de ejemplos]
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            loss = model(**batch).loss
        n = batch["input_ids"].size(0)
        stats += torch.tensor([loss.item() * n, n], device=device)
    dist.all_reduce(stats)
    model.train()
    return (stats[0] / stats[1].clamp(min=1)).item()


def main():
    args = parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    is_main = rank == 0

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)

    # misma semilla en todos los ranks -> mismos pesos iniciales (DDP además los difunde)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    use_bf16 = args.precision == "bf16"

    tokenizer = AutoTokenizer.from_pretrained(common.MODEL_NAME)
    # el rank 0 tokeniza y escribe la caché en $LUSTRE (compartida entre nodos);
    # el resto espera y la lee, para evitar escrituras concurrentes
    if is_main:
        common.build_datasets(tokenizer, args.max_len, args.stride,
                              args.max_train_samples, args.max_eval_samples,
                              num_proc=args.num_workers)
    dist.barrier()
    train_ds, val_ds = common.build_datasets(
        tokenizer, args.max_len, args.stride,
        args.max_train_samples, args.max_eval_samples, num_proc=args.num_workers)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank,
                                       shuffle=True, seed=args.seed, drop_last=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, sampler=val_sampler,
                            num_workers=args.num_workers, pin_memory=True)

    model = AutoModelForQuestionAnswering.from_pretrained(common.MODEL_NAME).to(device)
    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, fused=True)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    global_batch = args.batch_size * world_size
    if is_main:
        print(f"[{args.tag}] world_size={world_size} nodes={os.environ.get('SLURM_NNODES', '1')} "
              f"gpu={torch.cuda.get_device_name(local_rank)} features_train={len(train_ds)} "
              f"batch/GPU={args.batch_size} global_batch={global_batch} "
              f"steps/epoch={len(train_loader)} precision={args.precision}", flush=True)

    model.train()
    epoch_times, losses = [], []
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)  # baraja distinto en cada época
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        running, steps = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                loss = model(**batch).loss
            loss.backward()  # all-reduce de gradientes solapado con el backward
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            running += loss.item()
            steps += 1
        torch.cuda.synchronize()
        dist.barrier()
        dt = time.perf_counter() - t0
        epoch_times.append(dt)
        # loss media del epoch, promediada entre ranks
        l = torch.tensor(running / steps, device=device)
        dist.all_reduce(l)
        losses.append((l / world_size).item())
        if is_main:
            print(f"[{args.tag}] epoch {epoch + 1}/{args.epochs} train_loss={losses[-1]:.4f} "
                  f"time={dt:.1f}s throughput={steps * global_batch / dt:.1f} samples/s",
                  flush=True)

    vloss = eval_loss(model, val_loader, device, use_bf16)
    peak = torch.tensor(torch.cuda.max_memory_allocated() / 1e9, device=device)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)

    if is_main:
        train_time = sum(epoch_times)
        result = {
            "tag": args.tag, "gpus": world_size,
            "nodes": int(os.environ.get("SLURM_NNODES", 1)),
            "batch_per_gpu": args.batch_size, "global_batch": global_batch,
            "epochs": args.epochs, "precision": args.precision,
            "train_features": len(train_ds), "epoch_times_s": epoch_times,
            "train_time_s": train_time,
            "throughput_samples_s": len(train_loader) * global_batch * args.epochs / train_time,
            "train_loss": losses, "val_loss": vloss,
            "peak_mem_gb_max_rank": peak.item(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
        print("RESULT " + json.dumps(result), flush=True)
        common.RUNS_DIR.mkdir(parents=True, exist_ok=True)
        out = common.RUNS_DIR / f"{args.tag}_{os.environ.get('SLURM_JOB_ID', 'local')}.json"
        out.write_text(json.dumps(result, indent=2))
        print("Resultados en", out)
        if args.save_model:
            model.module.save_pretrained(common.RUNS_DIR / f"{args.tag}_{os.environ.get('SLURM_JOB_ID', 'local')}_model")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
