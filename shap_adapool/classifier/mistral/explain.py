import torch
import shap
import scipy as sp
import numpy as np
import pickle
from torch import nn
from .init import set_up_model_and_tokenizer
from rich.console import Console
from functools import partial
from .fine_tune import prepare_dataset_splits, tokenize, test
from ...datasets.open_canada.hf_dataset import load_split
from ...datasets.open_canada.get_data import get_data
from ...plotting import save_plot


def main():
    # TODO: take logic from here and the `fine_tune.py` and put it in a separate file to set up the model (increase code reuse)
    console = Console()
    model, tokenizer = set_up_model_and_tokenizer(checkpoint="results/model/checkpoint-500")

    def f(x):
        tv = torch.tensor([tokenizer.encode(v, padding='max_length', max_length=500, truncation=True) for v in x]).cuda()
        outputs = model(tv)[0].detach()
        scores = nn.Softmax(dim=-1)(outputs).detach().cpu().numpy().astype(np.float64)
        return scores

    # build an explainer using a token masker
    explainer = shap.Explainer(f, tokenizer)

    hf_dataset = load_split()
    tokenized_dataset = hf_dataset.map(partial(tokenize, tokenizer=tokenizer), batched=True)
    shap_values = explainer(tokenized_dataset["test"]['text'])

    with open("results/shap_values.pkl", "wb") as f:
        pickle.dump(shap_values, f)

    # Optional: save a veeeery big original plot collection:
    # plot = shap.plots.text(shap_values, display=False)
    # save_plot(plot, "shap_plot_mistral")

    console.print("Done")


if __name__ == "__main__":
    main()
