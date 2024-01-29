# SHAP Adaptive Pooling

## Installation

## Usage

### Sentence Pooling

The most naive pipeline is taking all the generated Shapley Values and pooling them together sentence-by-sentence. The pipeline with the provided example file `shap.pkl` can be run as follows:

```bash
python -m shap_adapool.pooling_strategies.sentence_pooling
```
