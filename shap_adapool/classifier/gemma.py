import torch
import numpy as np
from torch import nn
from pathlib import Path
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

from ..datasets.open_canada.hf_dataset import create_hf_dataset, train_val_test_split
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


def fine_tune(model, tokenizer, dataset: Dataset):

    def tokenize(dataset: Dataset):
        return tokenizer(dataset["text"],
                         truncation=True)

    split_dataset = train_val_test_split(dataset)

    tokenized_dataset = split_dataset.map(tokenize, batched=True)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    compute_metrics = make_metrics_func(split_dataset)

    model_save_dir = Path("results/model")
    training_args = TrainingArguments(model_save_dir,
                                      evaluation_strategy="epoch")

    # Quantized models require a LoRA configuration:
    lora_config = LoraConfig(r=16,
                             lora_alpha=32,
                             lora_dropout=0.05,
                             bias="none")
    model = get_peft_model(model, lora_config)

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics
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
    num_labels = 2
    # model_id = "google/gemma-7b"
    model_id = "mistralai/Mistral-7B-v0.1"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id,
                                               quantization_config=bnb_config,
                                               device_map={"":0},
                                               num_labels=num_labels)

    # We need to explicitly set the padding token, one way to do this is to set it to EOS token:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_eos_token = True
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.eos_token_id
    # It's kind of disgusting, the code above, I really hate it

    df = get_data()
    hf_dataset = create_hf_dataset(df)
    model = fine_tune(model, tokenizer, hf_dataset)
    test(model, tokenizer)


if __name__ == "__main__":
    main()
