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

## Results (latest run)

On the public Adult dataset (single 75/25 split), a class-balanced LightGBM with isotonic calibration reduced test Brier score to 0.087 from 0.101 for a standard 300-tree random forest using raw probabilities (lower is better), a gain from one split only.

| Metric | Value |
|---|---|
| baseline_score | 0.1006 |
| improved_score | 0.08694 |
| improvement | 0.01366 |
| baseline_score_std | 0.00008303 |
| improved_score_std | 0 |
| baseline_f1 | 0.7933 |
| improved_f1 | 0.818 |
| baseline_accuracy | 0.8573 |
| improved_accuracy | 0.8758 |
| n_model_seeds | 3 |

Run on 2026-10-20. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Single data split; spread reflects model seeds only, not split variability. Hyperparameters were fixed a priori with no search, so the test set was not used for selection. fetch_openml does not accept a timeout argument, so only the reachability check uses timeout=60; fetch_openml itself is retried on errors. OpenML Adult is not the paper's dataset; the paper's own model and result are not stated, so this is a generic illustration of calibration, not a replication. Class weighting distorts probabilities; isotonic calibration on the training folds is intended to correct that. Macro-F1 and accuracy use a 0.5 threshold and are secondary.

## Reference

This project is motivated by the following paper:

> A Maedche, C Legner, S Morana, A Benlian (2019). *AI-based digital assistants-opportunities, threats, and research perspectives*. … & information systems. https://aisel.aisnet.org/cgi/viewcontent.cgi?article=1532&context=bise

Paper author: Prof. Christine Legner, University of Lausanne (christine.legner@unil.ch)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
