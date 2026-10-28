"""Utilidades compartidas: rutas, tokenización de SQuAD y construcción de datasets.

Datos, caché de HuggingFace y resultados viven en $LUSTRE/hpctools (nunca en
$HOME ni en el repo), con fallback a una carpeta local para probar en un portátil.
"""
import os
from pathlib import Path

MODEL_NAME = "google-bert/bert-base-uncased"
DATASET_NAME = "rajpurkar/squad"
COLUMNS = ["input_ids", "token_type_ids", "attention_mask", "start_positions", "end_positions"]


def base_dir() -> Path:
    lustre = os.environ.get("LUSTRE")
    return Path(lustre) / "hpctools" if lustre else Path(__file__).resolve().parent / "_local"


def setup_hf_env() -> None:
    """Redirige la caché de HF a $LUSTRE (llamar antes de importar datasets/transformers)."""
    hf = base_dir() / "hf"
    os.environ.setdefault("HF_HOME", str(hf))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf / "datasets"))


RUNS_DIR = base_dir() / "runs"


def prepare_features(examples, tokenizer, max_len, stride):
    """Receta estándar de HF para QA extractiva: ventanas con solape y posiciones start/end."""
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_len,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_map = tokenized.pop("overflow_to_sample_mapping")
    offsets_all = tokenized.pop("offset_mapping")
    tokenized["start_positions"] = []
    tokenized["end_positions"] = []

    for i, offsets in enumerate(offsets_all):
        input_ids = tokenized["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        seq_ids = tokenized.sequence_ids(i)
        answers = examples["answers"][sample_map[i]]

        if len(answers["answer_start"]) == 0:
            tokenized["start_positions"].append(cls_index)
            tokenized["end_positions"].append(cls_index)
            continue

        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        tok_start = 0
        while seq_ids[tok_start] != 1:
            tok_start += 1
        tok_end = len(input_ids) - 1
        while seq_ids[tok_end] != 1:
            tok_end -= 1

        if not (offsets[tok_start][0] <= start_char and offsets[tok_end][1] >= end_char):
            # la respuesta no cae en esta ventana
            tokenized["start_positions"].append(cls_index)
            tokenized["end_positions"].append(cls_index)
        else:
            while tok_start < len(offsets) and offsets[tok_start][0] <= start_char:
                tok_start += 1
            tokenized["start_positions"].append(tok_start - 1)
            while offsets[tok_end][1] >= end_char:
                tok_end -= 1
            tokenized["end_positions"].append(tok_end + 1)
    return tokenized


def build_datasets(tokenizer, max_len, stride, max_train_samples=None,
                   max_eval_samples=None, num_proc=8):
    """Devuelve (train, validation) ya tokenizados y en formato torch."""
    from datasets import load_dataset

    raw = load_dataset(DATASET_NAME)
    train_raw, val_raw = raw["train"], raw["validation"]
    if max_train_samples:
        train_raw = train_raw.select(range(min(max_train_samples, len(train_raw))))
    if max_eval_samples:
        val_raw = val_raw.select(range(min(max_eval_samples, len(val_raw))))

    def fn(batch):
        return prepare_features(batch, tokenizer, max_len, stride)

    train = train_raw.map(fn, batched=True, remove_columns=train_raw.column_names,
                          num_proc=num_proc, desc="tokenize train")
    val = val_raw.map(fn, batched=True, remove_columns=val_raw.column_names,
                      num_proc=num_proc, desc="tokenize validation")
    return train.with_format("torch", columns=COLUMNS), val.with_format("torch", columns=COLUMNS)
