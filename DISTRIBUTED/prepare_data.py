"""Descarga modelo, tokenizador y SQuAD a la caché de $LUSTRE.

Ejecutar UNA vez en el nodo de login (los nodos de cómputo pueden no tener
internet); los jobs corren después con HF_HUB_OFFLINE=1.
"""
import common

common.setup_hf_env()

from datasets import load_dataset  # noqa: E402
from transformers import AutoModelForQuestionAnswering, AutoTokenizer  # noqa: E402

AutoTokenizer.from_pretrained(common.MODEL_NAME)
AutoModelForQuestionAnswering.from_pretrained(common.MODEL_NAME)
ds = load_dataset(common.DATASET_NAME)
print(ds)
print("Caché en", common.base_dir() / "hf")
