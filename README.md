# SHAP Adaptive Pooling

This project explores the possibilities of Shapley Value pooling strategies. Since Shapley Values are additive in nature, it is possible to combine input tokens into less granular phrases and sum up the corresponding Shapley Values to obtain phrase-level explanations. Of course the semantic definition of a _phrase_ is loosely defined as a grouping of words that have some syntactic function in a given context.

We explore different strategies of pooling Shapley Values into:

- [x] sentences
- [ ] k-word phrases
- [ ] language-syntax-tree-defined phrases
- [ ] adaptively-defined phrases

## Installation

This project uses the [Poetry Package Manager](https://python-poetry.org/) and the recommended way to install the project is to:

1. Build the package: `poetry build`
2. Find the wheel in the `dist` folder and install the wheel with `pip`: `pip install <path-to-wheel>`

> [!warning]
> At least Python 3.10 is required for this package to work. We profusely use functional programming concepts such as structural pattern matching and some of these facilities are only available in Python 3.10 and newer.

> [!warning]
> If you're developing the project, install with `poetry install` instead.

## Usage

### Sentence Pooling

The most naive pipeline is taking all the generated Shapley Values and pooling them together sentence-by-sentence. The pipeline with the provided example file `shap.pkl` can be run as follows:

```bash
python -m shap_adapool.pooling_strategies.sentence_pooling
```

### k-Word Pooling

TODO

### LST Pooling

TODO

### Adaptive Pooling

TODO

## Development

For development:

1. Clone this repo.
2. Install the repo using `poetry install`. This will install the package in editable mode.
