import matplotlib
import matplotlib.pyplot as plt


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
