import torch
import shap
import scipy as sp
import numpy as np
import pickle
import argparse
from torch import nn
from .init import set_up_model_and_tokenizer
from rich.console import Console
from functools import partial
from .fine_tune import prepare_dataset_splits, tokenize, test
from ...datasets.open_canada.hf_dataset import load_split
from ...datasets.open_canada.get_data import get_data
from ...datasets.open_canada.data_source import DATASET_OUTPUT_PATH
from ...datasets.ag_news.data_source import DATASET_OUTPUT_PATH as DATASET_OUTPUT_PATH_AG
from ...plotting import save_plot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset",
                        type=str,
                        choices=["ag_news", "open_canada"],
                        help="Dataset to run the explainer for")
    parser.add_argument("-l", "--limit",
                        type=int,
                        help="Max number of the samples to be explained",
                        default=-1)
    args = parser.parse_args()

    console = Console()
    model, tokenizer = set_up_model_and_tokenizer(checkpoint="results/model/checkpoint-500")

    def f(x):
        tv = torch.tensor([tokenizer.encode(v, padding='max_length', max_length=500, truncation=True) for v in x]).cuda()
        outputs = model(tv)[0].detach()
        scores = nn.Softmax(dim=-1)(outputs).detach().cpu().numpy().astype(np.float64)
        return scores

    # build an explainer using a token masker
    explainer = shap.Explainer(f, tokenizer)

    match args.dataset:
        case "ag_news":
            hf_dataset = load_split(DATASET_OUTPUT_PATH_AG)
        case "open_canada":
            hf_dataset = load_split(DATASET_OUTPUT_PATH)

    tokenized_dataset = hf_dataset.map(partial(tokenize, tokenizer=tokenizer), batched=True)

    if args.limit > 0:
        tokenized_dataset_to_explain = tokenized_dataset["test"].select(range(args.limit))
    else:
        tokenized_dataset_to_explain = tokenized_dataset["test"]

    shap_values = explainer(tokenized_dataset_to_explain['text'])

    with open("results/shap_values.pkl", "wb") as f:
        pickle.dump(shap_values, f)

    # Optional: save a veeeery big original plot collection:
    plot = shap.plots.text(shap_values, display=False)
    save_plot(plot, "shap_plot_mistral")

    console.print("Done")


if __name__ == "__main__":
    main()
