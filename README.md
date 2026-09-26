# Spam-Detection-for-YouTube-Video-Comments
Replication and extension of Xiao Liang (2024) for YouTube comment spam detection using machine learning models on the UCI dataset.
# Spam Detection for YouTube Video Comments

**Course Project — CSE 477 (Data Mining)**
East West University, Dept. of Computer Science & Engineering


## Overview

This project replicates the ML pipeline from Xiao & Liang (2024) for detecting spam in YouTube comments, using the **UCI YouTube Spam Collection** dataset (1,956 comments from 5 viral music videos: Psy, Katy Perry, LMFAO, Eminem, Shakira). After deduplication and preprocessing, 1,890 usable comments remained.

Eight classifiers were trained and evaluated: **Gaussian Naive Bayes, Logistic Regression, KNN (k=5), SVM (RBF), MLP, Random Forest, Decision Tree**, and a **soft-voting Voting Classifier**.

### Original Contributions

1. **Extended evaluation metrics** — added **Macro F1** and **per-class Recall**, which are absent from the original paper. This exposed that KNN, despite a perfect Precision of 1.0000, correctly detects only ~32.6% of actual spam (Macro F1 = 0.619) — a failure invisible under precision-only reporting.
2. **N-gram comparison** — a systematic unigram (1,1) vs. bigram (1,2) TF-IDF comparison across all 8 classifiers, yielding a **null result**: the Voting Classifier's Macro F1 stayed at 0.9418 in both configurations, showing spam signal in this dataset comes from high-signal individual words ("check", "subscribe", "channel") rather than multi-word phrases.

## Methodology

1. **Data loading & integrity checks** — 5 CSVs merged, source video preserved as a column
2. **Text preprocessing** — lowercasing, URL removal, HTML tag stripping, special character removal, whitespace normalization
3. **Feature extraction** — TF-IDF vectorization (English stop-words removed, max 5,000 features), unigram and bigram configurations
4. **Train/test split** — 80/20 stratified split (1,512 train / 378 test)
5. **Classifier training** — 8 models trained on both n-gram configurations
6. **Evaluation** — Accuracy, Precision, Recall (per-class), Macro F1, AUC-ROC, ROC curves, confusion matrices

## Key Results

**Paper replication (Unigram TF-IDF):**

| Model | Accuracy | Precision | AUC-ROC |
|---|---|---|---|
| **Voting Classifier** | 0.9418 | 0.9667 | **0.9785** |
| Random Forest | 0.9339 | 0.9661 | 0.9779 |
| SVM (RBF) | 0.9259 | 0.9655 | 0.9710 |
| Logistic Regression | 0.9312 | 0.9713 | 0.9688 |
| MLP | 0.9021 | 0.9474 | 0.9529 |
| Decision Tree | 0.9206 | 0.9396 | 0.9441 |
| KNN (k=5) | 0.6640 | 1.0000 | 0.8575 |
| Gaussian NB | 0.7382 | 0.8138 | 0.7387 |

- The Voting Classifier is the top performer overall, confirming the original paper's finding that ensembles outperform individual classifiers.
- Our Random Forest AUC-ROC (0.9779, single holdout split) is consistent with the paper's reported 0.984 (cross-validation).
- **KNN's precision/recall paradox:** perfect precision (1.0000) but Spam Recall of only 0.3263 — it misses ~67% of actual spam, a flaw invisible without Macro F1.
- **Bigram extension:** no meaningful improvement over unigrams (mean Macro F1 change of +0.0027 across all models), confirming that single high-signal words dominate spam detection in this dataset.

## Repository Structure

```
.
├── data/                   # UCI YouTube Spam Collection CSVs (5 files)
├── notebooks/              # Analysis notebook(s)
├── figures/                # EDA plots, ROC curves, confusion matrices
├── report/                 # Full project report (PDF)
└── README.md
```
*(Adjust this section to match your actual repository layout.)*

## Dataset

**UCI YouTube Spam Collection** — 1,956 comments across 5 CSV files (one per video), each containing comment text and a binary spam/ham label.
Source: https://archive.ics.uci.edu/ml/datasets/YouTube+Spam+Collection

## Tech Stack

- Python
- scikit-learn (TF-IDF, classifiers, evaluation metrics)
- pandas / NumPy
- matplotlib (EDA, ROC curves, confusion matrices)

## Reference

L. Xiao and X. Liang, "Traditional machine learning approaches for YouTube spam detection," *Machine Learning with Applications*, vol. 16, p. 100550, 2024. DOI: [10.1016/j.mlwa.2024.100550](https://doi.org/10.1016/j.mlwa.2024.100550)

## License

*(Add a license if applicable, e.g. MIT)*
