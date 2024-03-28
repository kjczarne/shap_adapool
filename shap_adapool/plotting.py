import matplotlib
import matplotlib.pyplot as plt
import pickle
import shap
import argparse
from pathlib import Path


def save_plot(plot: matplotlib.figure.Figure | str,
              plot_name: str,
              directory: str = "results"):
    """Saves a plot to the results directory."""
    name = plot_name.replace(" ", "_").lower()
    match plot:
        case str(plot):
            with open(f"{directory}/shap_plot_{name}.html", "w") as f:
                f.write(plot)
        case fig if isinstance(fig, matplotlib.figure.Figure):
            plot.savefig(f"{directory}/shap_plot_{name}.png")
        case _:
            raise ValueError(f"Unknown plot type {type(plot)}")


def main():
    parser = argparse.ArgumentParser("Create a SHAP plot from stored SHAP values.")
    parser.add_argument("-p", "--path", type=str, help="Path to the shap_values.pkl file",
                        default="results/shap_values.pkl")

    args = parser.parse_args()
    path = Path(args.path)

    with open(path, "rb") as f:
        shap_values = pickle.load(f)

    # Optional: save a veeeery big original plot collection:
    plot = shap.plots.text(shap_values, display=False)
    save_plot(plot, "shap_plot_mistral")


if __name__ == "__main__":
    main()
