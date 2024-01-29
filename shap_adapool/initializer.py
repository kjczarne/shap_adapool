from pathlib import Path


def init():
    """Bootstraps the project runs"""

    # 1. create a results directory if it doesn't exist:
    Path("results").mkdir(exist_ok=True)
