# Deep Neural Networks - Programming Assignment 1

## Assignment Summary

This assignment implements and compares a **baseline linear model** (Logistic Regression)
and a **Multi-Layer Perceptron (MLP)** — both built entirely from scratch using only NumPy —
on a real-world tabular dataset, then evaluates and analyzes their performance.

## Dataset

**Breast Cancer Wisconsin (Diagnostic)**
- Source: UCI ML Repository (loaded via `sklearn.datasets`)
- Samples: 569
- Features: 30 numeric diagnostic measurements
- Problem: Binary classification (Malignant vs Benign)
- Primary metric: **Recall** — in medical diagnosis, missing cancer (false negative) is
  far more costly than a false alarm.

## Project Structure

```
DNN-Assignment-1/
├── README.md
├── requirements.txt
└── assignment.py          # Full implementation (dataset, models, evaluation, plots)
```

## How to Run

```bash
pip install -r requirements.txt
python assignment.py
```

Outputs:
- Training loss curves saved to `loss_curves.png`
- Performance comparison bar chart saved to `performance_comparison.png`
- All metrics and analysis printed to stdout

## Implementation Details

| Component | Description |
|-----------|-------------|
| Baseline  | Logistic Regression with gradient descent (from scratch) |
| MLP       | 2 hidden layers, ReLU activations, sigmoid output, backpropagation (from scratch) |
| Allowed libs | NumPy, Pandas, Matplotlib, Seaborn, sklearn (only for split/scaling/encoding) |
