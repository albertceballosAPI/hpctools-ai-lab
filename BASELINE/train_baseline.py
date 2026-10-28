"""BASELINE: fine-tuning de BERT-base en SQuAD v1.1 con 1 GPU (PyTorch puro).

Bucle de entrenamiento explícito (sin Trainer) para controlar qué se mide:
el tiempo reportado es solo el de entrenamiento (sin tokenización ni carga),
con cuda.synchronize() al inicio y fin de cada época.
"""
import argparse
import json
import os
import time

import common

common.setup_hf_env()

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
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
    p.add_argument("--max-train-samples", type=int, default=None,
                   help="limita ejemplos (prueba rápida)")
    p.add_argument("--max-eval-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default="baseline")
    return p.parse_args()


@torch.no_grad()
def eval_loss(model, loader, device, use_bf16):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            loss = model(**batch).loss
        total += loss.item() * batch["input_ids"].size(0)
        n += batch["input_ids"].size(0)
    model.train()
    return total / max(n, 1)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    use_bf16 = args.precision == "bf16"

    tokenizer = AutoTokenizer.from_pretrained(common.MODEL_NAME)
    train_ds, val_ds = common.build_datasets(
        tokenizer, args.max_len, args.stride,
        args.max_train_samples, args.max_eval_samples, num_proc=args.num_workers)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = AutoModelForQuestionAnswering.from_pretrained(common.MODEL_NAME).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, fused=True)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    print(f"[{args.tag}] gpu={torch.cuda.get_device_name(0)} features_train={len(train_ds)} "
          f"batch={args.batch_size} steps/epoch={len(train_loader)} precision={args.precision}",
          flush=True)

    model.train()
    epoch_times, losses = [], []
    for epoch in range(args.epochs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        running, steps = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            running += loss.item()
            steps += 1
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        epoch_times.append(dt)
        losses.append(running / steps)
        print(f"[{args.tag}] epoch {epoch + 1}/{args.epochs} train_loss={losses[-1]:.4f} "
              f"time={dt:.1f}s throughput={steps * args.batch_size / dt:.1f} samples/s",
              flush=True)

    vloss = eval_loss(model, val_loader, device, use_bf16)
    train_time = sum(epoch_times)
    result = {
        "tag": args.tag, "gpus": 1, "nodes": 1,
        "batch_per_gpu": args.batch_size, "global_batch": args.batch_size,
        "epochs": args.epochs, "precision": args.precision,
        "train_features": len(train_ds), "epoch_times_s": epoch_times,
        "train_time_s": train_time,
        "throughput_samples_s": len(train_loader) * args.batch_size * args.epochs / train_time,
        "train_loss": losses, "val_loss": vloss,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    print("RESULT " + json.dumps(result), flush=True)

    common.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = common.RUNS_DIR / f"{args.tag}_{os.environ.get('SLURM_JOB_ID', 'local')}.json"
    out.write_text(json.dumps(result, indent=2))
    print("Resultados en", out)


if __name__ == "__main__":
    main()
