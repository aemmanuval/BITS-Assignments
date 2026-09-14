# Deep Neural Networks - Programming Assignment 1

## Assignment Summary

This assignment implements and compares a **baseline linear model** (Logistic Regression)
and a **Multi-Layer Perceptron (MLP)** — both built entirely from scratch using only NumPy —
on a real-world tabular dataset, then evaluates and analyzes their performance.

## Dataset

**Online Shoppers Purchasing Intention**
- Source: UCI ML Repository
- Samples: 12,330
- Features: 17 (page visits, durations, bounce/exit rates, visitor info, etc.)
- Problem: Binary classification (Purchase vs No Purchase)
- Primary metric: **F1-Score** — dataset is imbalanced (84.5% non-purchase), so accuracy
  alone is misleading. F1 balances precision and recall.

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
| MLP       | 2 hidden layers (64, 32), ReLU activations, sigmoid output, backpropagation (from scratch) |
| Allowed libs | NumPy, Pandas, Matplotlib, Seaborn, sklearn (only for split/scaling/encoding) |
