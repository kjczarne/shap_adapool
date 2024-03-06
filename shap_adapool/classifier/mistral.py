import torch
import numpy as np
from torch.nn import functional as F
from torch import nn
from pathlib import Path
from datetime import datetime
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          BitsAndBytesConfig,
                          GemmaTokenizer,
                          TrainingArguments,
                          AutoModelForSequenceClassification,
                          Trainer,
                          DataCollatorWithPadding)
import evaluate
from datasets import Dataset
from peft import LoraConfig, get_peft_model
# from trl import SFTTrainer

from ..datasets.open_canada.hf_dataset import create_hf_dataset, train_val_test_split, TOP_CLASSES
from ..datasets.open_canada.get_data import get_data

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
            labels = F.one_hot(labels, num_classes=self.model.config.num_labels)

        loss = F.binary_cross_entropy_with_logits(logits,
                                                labels.to(torch.float32))
        return (loss, outputs) if return_outputs else loss


def fine_tune(model, tokenizer, dataset: Dataset, with_lora: bool = True):

    def tokenize(dataset: Dataset):
        return tokenizer(dataset["text"],
                         truncation=True,
                         max_length=260,
                         padding="max_length")

    split_dataset = train_val_test_split(dataset)

    tokenized_dataset = split_dataset.map(tokenize, batched=True)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    compute_metrics = make_metrics_func(split_dataset)

    model_save_dir = Path("results/model")

    run_name = "my-mom-doesnt-love-me"

    if with_lora:
        model = get_qlora_model(model)

    trainer = MulticlassTextClassificationTrainer(
        model=model,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"],
        compute_metrics=compute_metrics,
        args=TrainingArguments(
            output_dir=model_save_dir,
            warmup_steps=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            # gradient_checkpointing=True,
            max_steps=500,
            learning_rate=2.5e-5, # Want a small lr for finetuning
            bf16=True,
            optim="paged_adamw_8bit",
            logging_steps=25,              # When to start reporting loss
            logging_dir="./logs",        # Directory for storing logs
            save_strategy="steps",       # Save the model checkpoint every logging step
            save_steps=25,                # Save checkpoints every 50 steps
            evaluation_strategy="steps", # Evaluate the model every logging step
            eval_steps=25,               # Evaluate and save checkpoints every 50 steps
            do_eval=True,                # Perform evaluation at the end of training
            # report_to="wandb",           # Comment this out if you don't want to use weights & baises
            run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
        ),
        # data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        data_collator=collator
    )
    trainer.train()

    return model, tokenizer


def test(model, tokenizer):
    text1 = """This company is a producer of processed food items such as canned legumes,
canned vegetables, and canned fruits. It is headquartered in the United States."""
    encoded_input = tokenizer(text1, return_tensors='pt')
    device = "cuda:0"
    input_ids = encoded_input['input_ids'].to(device)
    attn_mask = encoded_input['attention_mask'].to(device)
    output = model(input_ids=input_ids, attention_mask=attn_mask)

    softmax_output = nn.Softmax(dim=-1)(output.logits)
    print(softmax_output)


def main():
    num_labels = 4
    # model_id = "google/gemma-7b"
    model_id = "mistralai/Mistral-7B-v0.1"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id,
                                               quantization_config=bnb_config,
                                               device_map={"":0},
                                               num_labels=num_labels,
                                               trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(model_id,
    #                                              quantization_config=bnb_config,
    #                                              device_map={"":0},
    #                                              trust_remote_code=True)

    # We need to explicitly set the padding token, one way to do this is to set it to EOS token:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_eos_token = True
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.eos_token_id
    # It's kind of disgusting, the code above, I really hate it

    df = get_data()
    hf_dataset = create_hf_dataset(df, TOP_CLASSES)
    model = fine_tune(model, tokenizer, hf_dataset)
    test(model, tokenizer)
    print("Done")


if __name__ == "__main__":
    main()
