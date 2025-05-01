import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, HfArgumentParser, TrainingArguments
from datasets import load_dataset
from dataclasses import dataclass, field

MODEL_NAME = "bert-base-uncased"
DATASET_NAME = "nyu-mll/glue"
TASK_NAME = "mrpc"


@dataclass
class MyArguments:
    max_train_samples: int = field(default=-1, metadata={"help": "Limit train samples (–1 = all)"})
    max_eval_samples: int = field(default=-1, metadata={"help": "Limit eval samples (–1 = all)"})
    max_predict_samples: int = field(default=-1, metadata={"help": "Limit predict samples (–1 = all)"})
    do_train: bool = field(default=False, metadata={"help": "Whether to run training"})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run prediction"})
    model_path: str = field(default=None, metadata={"help": "Path to fine-tuned model for prediction"})


def parse_args():
    parser = HfArgumentParser((MyArguments, TrainingArguments))
    my_args, training_args = parser.parse_args_into_dataclasses()


def load_model_and_dataset():
    """
    Loads the model (with the relevant tokenizer) and dataset from HuggingFace.
    :return: tokenizer, model, dataset
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    ds = load_dataset(DATASET_NAME, TASK_NAME)
    return tokenizer, model, ds


def split_dataset(ds):
    """
    Splits dataset into train, validation and test sets.
    :param ds: The dataset to split.
    :return: train_set, val_set, test_set
    """
    train_set = ds['train']
    val_set = ds['validation']
    test_set = ds['test']
    return train_set, val_set, test_set
