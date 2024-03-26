import torch
import argparse
import numpy as np
from torch.nn import functional as F
from torch import nn
from pathlib import Path
from datetime import datetime
from typing import Callable
from transformers import (AutoTokenizer,
                          TrainingArguments,
                          Trainer,
                          DataCollatorWithPadding)
import evaluate
from datasets import Dataset, DatasetDict
from peft import LoraConfig, get_peft_model
from rich.console import Console
from functools import partial
from torch.utils.tensorboard import SummaryWriter
from rich.console import Console

from .init import set_up_model_and_tokenizer
from ...datasets.open_canada.hf_dataset import create_hf_dataset, train_val_test_split, TOP_CLASSES, save_split, load_split
from ...datasets.open_canada.get_data import get_data
from ...datasets.ag_news.hf_dataset import create_hf_dataset as create_hf_dataset_ag
from ...datasets.ag_news.data_source import DATASET_OUTPUT_PATH as DATASET_OUTPUT_PATH_AG
from ...datasets.open_canada.data_source import DATASET_OUTPUT_PATH

# TODO: import `init()`

def make_metrics_func(*dataset_load_args):
    def compute_metrics(eval_pred):
        # note: metric is implicitly part of a `Dataset` object
        metric = evaluate.load(*dataset_load_args)
        logits, labels = eval_pred
        pred_class = np.argmax(logits, axis=-1)  # take the max-scoring logit as the predicted class ID
        return metric.compute(predictions=pred_class,
                              references=labels)
    return compute_metrics


def get_qlora_model(model):
    lora_config = LoraConfig(r=16,
                             lora_alpha=32,
                             lora_dropout=0.05,
                             bias="none")
    return get_peft_model(model, lora_config)


class MulticlassTextClassificationTrainer(Trainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    # overload loss calculation method from the original `Trainer` class:
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        logits = outputs.get("logits")

        if logits.shape != inputs["input_ids"].shape:
            # assuming that if the shapes don't match,
            # the labels are not hot-encoded on input
            labels = F.one_hot(labels, num_classes=self.model.config.num_labels)  # pylint: disable=E1102

        pred_classes = nn.Softmax(dim=-1)(logits)
        # loss = F.cross_entropy(pred_classes,
        #                        labels.to(torch.float32))
        loss = F.binary_cross_entropy_with_logits(logits,
                                                  labels.to(torch.float32))

        return (loss, outputs) if return_outputs else loss

def prepare_dataset_splits(dataset: Dataset) -> DatasetDict:
    return train_val_test_split(dataset)


def tokenize(dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    return tokenizer(dataset["text"],
                     truncation=True,
                     max_length=260,
                     padding="max_length")


def fine_tune(model,
              tokenizer,
              tokenized_train_dataset: Dataset,
              tokenized_val_dataset: Dataset,
              with_lora: bool = True):

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    compute_metrics = make_metrics_func(tokenized_train_dataset)

    model_save_dir = Path("results/model")

    run_name = "my-mom-doesnt-love-me"

    if with_lora:
        model = get_qlora_model(model)

    trainer = MulticlassTextClassificationTrainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        compute_metrics=compute_metrics,
        args=TrainingArguments(
            output_dir=model_save_dir,
            warmup_steps=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            # gradient_checkpointing=True,  # IMPORTANT: with gradient checkpointing, backprop did not seem to work!
            max_steps=500,
            learning_rate=2.5e-5,           # lr for finetuning
            bf16=True,
            optim="paged_adamw_8bit",
            logging_steps=25,               # Logging interval
            logging_dir="results/logs",     # Directory for storing logs
            save_strategy="steps",          # Save the model checkpoint every logging step
            save_steps=25,                  # Save checkpoints every N steps
            evaluation_strategy="steps",    # Evaluate the model every logging step
            eval_steps=25,                  # Evaluate and save checkpoints every 50 steps
            do_eval=True,                   # Perform evaluation at the end of training
            # report_to="wandb",            # Comment this out if you don't want to use weights & baises
            report_to="tensorboard",        # Comment this out if you don't want to use tensorboard
            run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
        ),
        data_collator=collator
    )
    trainer.train()

    return model, tokenizer


def one_sentence_test(model, tokenizer):
    text1 = """This company is a producer of processed food items such as canned legumes,
canned vegetables, and canned fruits. It is headquartered in the United States."""
    encoded_input = tokenizer(text1, return_tensors='pt')
    device = "cuda:0"
    input_ids = encoded_input['input_ids'].to(device)
    attn_mask = encoded_input['attention_mask'].to(device)
    output = model(input_ids=input_ids, attention_mask=attn_mask)

    softmax_output = nn.Softmax(dim=-1)(output.logits)
    print(softmax_output)


def test(model, tokenized_test_dataset: Dataset):
    predicted = []
    gt = []
    console = Console()
    writer = SummaryWriter("results/logs")
    for sample in tokenized_test_dataset:
        device = "cuda:0"
        input_ids = torch.tensor(sample['input_ids']).unsqueeze(dim=0).to(device)
        attn_mask = torch.tensor(sample['attention_mask']).unsqueeze(dim=0).to(device)
        output = model(input_ids=input_ids, attention_mask=attn_mask)

        softmax_output = nn.Softmax(dim=-1)(output.logits)
        y_pred = torch.argmax(softmax_output, dim=-1)
        y_gt = torch.tensor(sample['labels']).unsqueeze(dim=0).to(device)
        console.print(f"Predicted: {y_pred}, Ground truth: {y_gt}\n")
        predicted.append(y_pred)
        gt.append(y_gt)

    predicted = torch.cat(predicted, dim=0)
    gt = torch.cat(gt, dim=0)
    acc = (predicted == gt).sum().item() / len(gt)
    console.print(f"Test (holdout) set accuracy: {acc}")
    writer.add_scalar("Test/Accuracy", acc, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str,
                        choices=["ag_news", "open_canada"],
                        default="open_canada",
                        help="Dataset to fine-tune with")
    parser.add_argument("--load-splits", action="store_true",
                        help="Whether to load back existing splits for the dataset")
    args = parser.parse_args()

    console = Console()

    model, tokenizer = set_up_model_and_tokenizer()
 
    match args.dataset:
        case "ag_news":
            hf_dataset = create_hf_dataset_ag()
            dataset_output_path = DATASET_OUTPUT_PATH_AG
        case "open_canada":
            df = get_data()
            hf_dataset = create_hf_dataset(df, TOP_CLASSES)
            dataset_output_path = DATASET_OUTPUT_PATH
        case _:
            raise ValueError(f"{args.dataset} is not a supported dataset choice!")

    console.print(f"[blue]Loaded: {args.dataset} dataset [/blue]")

    if args.load_splits:
        split_dataset = load_split(dataset_output_path)
    else:
        split_dataset = prepare_dataset_splits(hf_dataset)
        save_split(split_dataset, dataset_output_path)

    tokenized_dataset = split_dataset.map(partial(tokenize, tokenizer=tokenizer),
                                          batched=True)

    model, tokenizer = fine_tune(model,
                                 tokenizer,
                                 tokenized_dataset["train"],
                                 tokenized_dataset["val"])

    console.print("[green][bold]Fine-tuning complete[/bold][/green]")

    test(model, tokenized_dataset["test"])
    print("Done")


if __name__ == "__main__":
    main()
