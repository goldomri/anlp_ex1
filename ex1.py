import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset


MODEL_NAME = "bert-base-uncased"
DATASET_NAME = "nyu-mll/glue"
TASK_NAME = "mrpc"


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
ds = load_dataset(DATASET_NAME, TASK_NAME)