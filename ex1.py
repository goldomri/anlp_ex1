import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, HfArgumentParser, TrainingArguments
from datasets import load_dataset
from dataclasses import dataclass, field

MODEL_NAME = "bert-base-uncased"
DATASET_NAME = "nyu-mll/glue"
TASK_NAME = "mrpc"


@dataclass
class DataTrainingArguments:
    max_train_samples: int = field(
        default=-1,
        metadata={"help": "Limit train samples (–1 = all)"}
    )
    max_eval_samples: int = field(
        default=-1,
        metadata={"help": "Limit eval samples (–1 = all)"}
    )
    max_predict_samples: int = field(
        default=-1,
        metadata={"help": "Limit predict samples (–1 = all)"}
    )
    model_path: str = field(
        default=None,
        metadata={"help": "Path to model for prediction"}
    )


@dataclass
class Hyperparameters:
    lr: float = field(
        default=1e-5,
        metadata={"help": "Learning rate"}
    )
    batch_size: int = field(
        default=30,
        metadata={"help": "Train batch size"}
    )


def parse_args():
    """
    Parse command line arguments.
    :return: my_args, training_args
    """
    parser = HfArgumentParser((DataTrainingArguments, TrainingArguments))
    data_args, training_args = parser.parse_args_into_dataclasses()


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


def main():
    tokenizer, model, ds = load_model_and_dataset()

    def preprocess_function(examples):
        result = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)
        return result
