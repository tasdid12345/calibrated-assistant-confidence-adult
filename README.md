# Calibrated LightGBM vs Random Forest Confidence on Adult Income

Digital assistants that answer questions or make recommendations often attach a confidence to each prediction, but raw confidence from common classifiers is frequently miscalibrated, which matters when an assistant acts on it. The paper "AI-based digital assistants-opportunities, threats, and research perspectives" (2019) discusses the opportunities and risks of such assistants, including the need for trustworthy outputs. This experiment takes a narrow, testable slice of that concern: whether the confidence values themselves can be made more trustworthy.

**Question:** Can a class-balanced, isotonic-calibrated LightGBM model give more trustworthy confidence for an assistant's income predictions than an uncalibrated random forest on the public Adult dataset?

**Dataset:** Adult income (OpenML data ID 1590, loaded with scikit-learn's `fetch_openml`). This is a public dataset and is not the paper's own data. The split is a single stratified 75/25 train/test split with a fixed seed.

**Baseline:** a standard 300-tree random forest on one-hot encoded features. Its `predict_proba` output is used directly as the assistant's confidence.

**Improved model:** the same preprocessing and data, with a class-balanced LightGBM classifier wrapped in 5-fold isotonic calibration, fitted only on the training split.

**Metric:** Brier score on the test split (lower is better). Macro-F1 and accuracy at the 0.5 threshold are reported as secondary metrics.

## How to run

```bash
pip install -r requirements.txt
python analysis.py
```

Results are written to `results.json`. Charts are saved to `figures/`.

## Reference

This project is motivated by the following paper:

> A Maedche, C Legner, S Morana, A Benlian (2019). *AI-based digital assistants-opportunities, threats, and research perspectives*. … & information systems. https://aisel.aisnet.org/cgi/viewcontent.cgi?article=1532&context=bise

Paper author: Prof. Christine Legner, University of Lausanne (christine.legner@unil.ch)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
