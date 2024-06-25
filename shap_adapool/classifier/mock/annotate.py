from pathlib import Path
import shap
import numpy as np
import pickle
import argparse
from functools import partial
from rich.console import Console
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from ..mistral.init import set_up_model_and_tokenizer
from ..mistral.fine_tune import tokenize
from rich.console import Console
from functools import partial
from ...datasets.split import load_split
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
    parser.add_argument("-o", "--output",
                        type=str,
                        help="Output path for the shap values",
                        default="results/shap_values.pkl")
    args = parser.parse_args()

    console = Console()
    # Replace this with the actual function to load your model and tokenizer
    _, tokenizer = set_up_model_and_tokenizer(checkpoint="results/model/checkpoint-500")

    # Replace this with the actual function to load your dataset
    if args.dataset == "ag_news":
        hf_dataset = load_split(DATASET_OUTPUT_PATH_AG)
    else:
        hf_dataset = load_split(DATASET_OUTPUT_PATH)

    tokenized_dataset = hf_dataset.map(partial(tokenize, tokenizer=tokenizer), batched=True)

    if args.limit > 0:
        tokenized_dataset_to_explain = tokenized_dataset["test"].select(range(args.limit))
    else:
        tokenized_dataset_to_explain = tokenized_dataset["test"]

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = html.Div([
        dcc.Store(id='store', data={'index': 0, 'annotations': []}),
        html.H1("SHAP Value Annotator"),
        html.Div(id='sample-display'),
        html.Div(id='token-inputs'),
        dbc.Button("Next Sample", id='next-sample', n_clicks=0),
        html.Div(id='done-message'),
    ])

    @app.callback(
        Output('sample-display', 'children'),
        Output('token-inputs', 'children'),
        Input('store', 'data')
    )
    def display_sample(data):
        """Updates the display to show the current sample and input fields for each token's SHAP value"""
        index = data['index']
        if index >= len(tokenized_dataset_to_explain):
            return "All samples annotated!", ""
        
        sample = tokenized_dataset_to_explain[index]
        tokens = tokenizer.convert_ids_to_tokens(sample['input_ids'])
        token_inputs = [html.Div([
            html.Label(f"Token: {token}"),
            dcc.Input(id={'type': 'token-input', 'index': i}, type='number', value=0, step=0.1)
        ]) for i, token in enumerate(tokens)]
        
        return f"Sample {index + 1}/{len(tokenized_dataset_to_explain)}: {' '.join(tokens)}", token_inputs

    @app.callback(
        Output('store', 'data'),
        Output('done-message', 'children'),
        Input('next-sample', 'n_clicks'),
        State('store', 'data'),
        State({'type': 'token-input', 'index': dash.dependencies.ALL}, 'value')
    )
    def update_annotations(n_clicks, data, values):
        """Updates the annotations store with the SHAP values entered for the current sample,
        increments the index to move to the next sample, and saves the annotations
        to a file when all samples have been annotated
        """
        if data['index'] >= len(tokenized_dataset_to_explain):
            return data, "Annotation completed."
        
        data['annotations'].append(values)
        data['index'] += 1
        
        if data['index'] >= len(tokenized_dataset_to_explain):
            with open(Path(args.output), "wb") as f:
                pickle.dump(data['annotations'], f)
            return data, "Annotation completed."
        
        return data, ""

    app.run_server(debug=True, use_reloader=False)


if __name__ == "__main__":
    main()
