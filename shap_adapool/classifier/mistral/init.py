import torch
from peft import PeftModel
from pathlib import Path
from rich.console import Console
from transformers import (AutoTokenizer,
                          BitsAndBytesConfig,
                          AutoModelForSequenceClassification)


def set_up_model_and_tokenizer(model_id="mistralai/Mistral-7B-v0.1",
                               num_labels=3,
                               checkpoint: str | Path | None = None):
    console = Console()
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

    if checkpoint is not None:
        model = PeftModel.from_pretrained(model, checkpoint)

    # We need to explicitly set the padding token, one way to do this is to set it to EOS token:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_eos_token = True
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.eos_token_id
    # It's kind of disgusting, the code above, I really hate it

    return model, tokenizer
