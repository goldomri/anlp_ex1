import numpy as np
import evaluate
import wandb
from datasets import load_dataset
from dataclasses import dataclass, field
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    EvalPrediction
)

MODEL_NAME = "bert-base-uncased"
DATASET_NAME = "nyu-mll/glue"
TASK_NAME = "mrpc"
PROJECT_DIR = "/content/gdrive/MyDrive/Colab Notebooks/anlp_ex1/"
RES_FILE = PROJECT_DIR + "res.txt"


################################################################################
# Command-line / dataclass interfaces                                          #
################################################################################

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
class HyperparameterArguments:
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
    :return: my_args, hyperparameters, training_args
    """
    parser = HfArgumentParser((DataTrainingArguments, HyperparameterArguments, TrainingArguments))
    data_args, hp_args, training_args, remaining_args = parser.parse_args_into_dataclasses(
        return_remaining_strings=True)
    return data_args, hp_args, training_args


################################################################################
# Load Dataset and Model helpers                                               #
################################################################################

def load_tokenizer_and_dataset():
    """
    Loads the model (with the relevant tokenizer) and dataset from HuggingFace.
    :return: tokenizer, raw_datasets
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    raw_datasets = load_dataset(DATASET_NAME, TASK_NAME)
    return tokenizer, raw_datasets


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


################################################################################
# Metric helper                                                                #
################################################################################

metric = evaluate.load("accuracy")


def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    return metric.compute(predictions=preds, references=p.label_ids)


################################################################################
# Main routine                                                                 #
################################################################################

def main():
    data_args, hp_args, training_args = parse_args()
    tokenizer, raw_datasets = load_tokenizer_and_dataset()

    def preprocess_function(examples):
        result = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)
        return result

    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
    train_set, val_set, test_set = split_dataset(tokenized_datasets)

    # sample limiting -----------------------------------------------------
    if data_args.max_train_samples > 0:
        train_set = train_set.select(range(data_args.max_train_samples))
    if data_args.max_eval_samples > 0:
        val_set = val_set.select(range(data_args.max_eval_samples))
    if data_args.max_predict_samples > 0:
        test_set = test_set.select(range(data_args.max_predict_samples))

    ########################################################################
    # Model & Trainer                                                      #
    ########################################################################
    if training_args.do_train:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    else:
        # in prediction-only mode we expect --model_path
        if not data_args.model_path:
            raise ValueError("--model_path must be supplied when --do_predict without --do_train.")
        model = AutoModelForSequenceClassification.from_pretrained(data_args.model_path)

    data_collator = DataCollatorWithPadding(tokenizer)

    # Weights & Biases ----------------------------------------------------
    training_args.report_to = ["wandb"]
    wandb.login()
    run_name = f"lr{hp_args.lr}_bs{hp_args.batch_size}_ep{training_args.num_train_epochs}"
    output_dir = PROJECT_DIR + run_name
    wandb.init(
        project="anlp_ex1",
        name=run_name,
        config={**vars(hp_args), **vars(data_args)},
    )

    training_args.output_dir = output_dir
    training_args.per_device_train_batch_size = hp_args.batch_size
    training_args.per_device_eval_batch_size = hp_args.batch_size
    training_args.learning_rate = hp_args.lr
    training_args.eval_strategy = "epoch"
    training_args.save_strategy = "no"
    training_args.logging_steps = 1

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set if training_args.do_train else None,
        eval_dataset=val_set,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    ########################################################################
    # Training & Validation                                                #
    ########################################################################

    if training_args.do_train:
        train_result = trainer.train()
        # save final checkpoint
        trainer.save_model(training_args.output_dir)
        # save metrics for res.txt ---------------------------------------
        metrics = trainer.evaluate()
        val_acc = metrics["eval_accuracy"]
        line = (f"epoch_num: {training_args.num_train_epochs}, "
                f"lr: {training_args.learning_rate}, "
                f"batch_size: {training_args.per_device_train_batch_size}, "
                f"eval_acc: {val_acc:.4f}\n")
        with open(RES_FILE, "a") as fp:
            fp.write(line)
        # train loss curve for train_loss.png -----------------------------
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()  # W&B captures loss at every step

    ########################################################################
    # Prediction                                                           #
    ########################################################################
    if training_args.do_predict:
        model.eval()
        preds = trainer.predict(test_set, metric_key_prefix="predict")
        test_raw = raw_datasets["test"]  # non-tokenised split
        with open(training_args.output_dir + "predictions.txt", "w", encoding="utf-8") as fp:
            for ex, label in zip(test_raw, np.argmax(preds.predictions, axis=1)):
                line = f"{ex['sentence1']}###{ex['sentence2']}###{label}"
                fp.write(line + "\n")

    wandb.finish()


if __name__ == '__main__':
    main()
