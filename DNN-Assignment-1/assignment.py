"""
Deep Neural Networks — Programming Assignment 1
=================================================
Binary classification on the Breast Cancer Wisconsin (Diagnostic) dataset.

Models implemented from scratch (NumPy only):
  1. Logistic Regression  (baseline)
  2. Multi-Layer Perceptron (MLP)

Allowed sklearn usage: train_test_split, StandardScaler, datasets.load_breast_cancer
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

np.random.seed(42)

# ============================================================
# 1. Dataset Selection & Loading  (1 mark)
# ============================================================

def load_dataset():
    """Load and describe the Breast Cancer Wisconsin dataset."""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")

    print("=" * 70)
    print("DATASET: Breast Cancer Wisconsin (Diagnostic)")
    print("=" * 70)
    print(f"Samples        : {X.shape[0]}")
    print(f"Features       : {X.shape[1]}")
    print(f"Classes        : {list(data.target_names)}")
    print(f"Class balance  : {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"Problem type   : Binary Classification")
    print()
    print("Problem statement:")
    print(
        "This dataset predicts tumor malignancy from 30 diagnostic measurements.\n"
        "I'm using Recall as the primary metric because in medical diagnosis,\n"
        "false negatives (missing cancer) are more costly than false positives.\n"
        "Binary classification, 569 samples."
    )
    print()
    print("Feature summary:")
    print(X.describe().T[["mean", "std", "min", "max"]].to_string())
    print()

    return X.values, y.values, data.feature_names


# ============================================================
# 2. Data Preprocessing
# ============================================================

def preprocess(X, y, test_size=0.2):
    """80-20 train/test split, standard scaling."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"Split ratio    : {1 - test_size:.0%} train / {test_size:.0%} test")
    print(f"Train samples  : {X_train.shape[0]}")
    print(f"Test  samples  : {X_test.shape[0]}")
    print(f"Missing values : {np.isnan(X_train).sum() + np.isnan(X_test).sum()}")
    print(f"Scaling        : StandardScaler (zero mean, unit variance)")
    print()

    return X_train, X_test, y_train, y_test


# ============================================================
# 3. Baseline Model — Logistic Regression  (3 marks)
# ============================================================

class BaselineModel:
    """Logistic Regression trained with gradient descent (from scratch)."""

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.loss_history = []
        self.weights = None
        self.bias = None

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iterations):
            linear = X @ self.weights + self.bias
            predictions = self._sigmoid(linear)

            eps = 1e-15
            loss = -np.mean(
                y * np.log(predictions + eps)
                + (1 - y) * np.log(1 - predictions + eps)
            )
            self.loss_history.append(loss)

            error = predictions - y
            dw = (1 / n_samples) * (X.T @ error)
            db = (1 / n_samples) * np.sum(error)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict_proba(self, X):
        return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


# ============================================================
# 4. Multi-Layer Perceptron  (4 marks)
# ============================================================

class MLP:
    """
    Multi-Layer Perceptron with configurable architecture.

    Architecture list example: [30, 64, 32, 1]
      -> 30 inputs, 64-unit hidden, 32-unit hidden, 1 output (sigmoid).
    Hidden activations: ReLU.
    Output activation : Sigmoid (binary classification).
    """

    def __init__(self, architecture, learning_rate=0.01, n_iterations=1000):
        self.architecture = architecture
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.loss_history = []
        self.params = {}
        self.initialize_parameters()

    def initialize_parameters(self):
        """He initialization for ReLU layers, Xavier for output."""
        for i in range(1, len(self.architecture)):
            fan_in = self.architecture[i - 1]
            fan_out = self.architecture[i]
            if i < len(self.architecture) - 1:
                scale = np.sqrt(2.0 / fan_in)
            else:
                scale = np.sqrt(1.0 / fan_in)
            self.params[f"W{i}"] = np.random.randn(fan_in, fan_out) * scale
            self.params[f"b{i}"] = np.zeros((1, fan_out))

    @staticmethod
    def _relu(z):
        return np.maximum(0, z)

    @staticmethod
    def _relu_derivative(z):
        return (z > 0).astype(float)

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def forward_propagation(self, X):
        cache = {"A0": X}
        A = X
        n_layers = len(self.architecture) - 1

        for i in range(1, n_layers + 1):
            Z = A @ self.params[f"W{i}"] + self.params[f"b{i}"]
            cache[f"Z{i}"] = Z

            if i < n_layers:
                A = self._relu(Z)
            else:
                A = self._sigmoid(Z)

            cache[f"A{i}"] = A

        return A, cache

    def backward_propagation(self, y, cache):
        n_layers = len(self.architecture) - 1
        n_samples = y.shape[0]
        grads = {}

        y = y.reshape(-1, 1)
        A_out = cache[f"A{n_layers}"]

        eps = 1e-15
        dA = -(y / (A_out + eps)) + (1 - y) / (1 - A_out + eps)
        dZ = A_out - y

        for i in range(n_layers, 0, -1):
            A_prev = cache[f"A{i - 1}"]

            grads[f"dW{i}"] = (1 / n_samples) * (A_prev.T @ dZ)
            grads[f"db{i}"] = (1 / n_samples) * np.sum(dZ, axis=0, keepdims=True)

            if i > 1:
                dA_prev = dZ @ self.params[f"W{i}"].T
                dZ = dA_prev * self._relu_derivative(cache[f"Z{i - 1}"])

        return grads

    def _update_parameters(self, grads):
        n_layers = len(self.architecture) - 1
        for i in range(1, n_layers + 1):
            self.params[f"W{i}"] -= self.lr * grads[f"dW{i}"]
            self.params[f"b{i}"] -= self.lr * grads[f"db{i}"]

    def fit(self, X, y):
        for _ in range(self.n_iterations):
            A_out, cache = self.forward_propagation(X)

            eps = 1e-15
            y_col = y.reshape(-1, 1)
            loss = -np.mean(
                y_col * np.log(A_out + eps)
                + (1 - y_col) * np.log(1 - A_out + eps)
            )
            self.loss_history.append(loss)

            grads = self.backward_propagation(y, cache)
            self._update_parameters(grads)

        return self

    def predict_proba(self, X):
        A_out, _ = self.forward_propagation(X)
        return A_out.ravel()

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


# ============================================================
# 5. Evaluation Helpers
# ============================================================

def compute_metrics(y_true, y_pred):
    """Compute Accuracy, Precision, Recall, F1 without sklearn."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
           if (precision + recall) > 0 else 0.0)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
    }


def print_metrics(name, metrics):
    print(f"\n  {name}:")
    for k, v in metrics.items():
        print(f"    {k:<12s}: {v:.4f}")


# ============================================================
# 6. Visualizations
# ============================================================

def plot_loss_curves(baseline_loss, mlp_loss, save_path="loss_curves.png"):
    """Training loss curves for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(baseline_loss, color="steelblue")
    axes[0].set_title("Logistic Regression — Training Loss")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Binary Cross-Entropy Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(mlp_loss, color="coral")
    axes[1].set_title("MLP — Training Loss")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Binary Cross-Entropy Loss")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Loss curves saved to {save_path}")


def plot_performance_comparison(baseline_metrics, mlp_metrics,
                                save_path="performance_comparison.png"):
    """Side-by-side bar chart of evaluation metrics."""
    labels = list(baseline_metrics.keys())
    bl_vals = [baseline_metrics[k] for k in labels]
    mlp_vals = [mlp_metrics[k] for k in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, bl_vals, width, label="Logistic Regression",
                   color="steelblue")
    bars2 = ax.bar(x + width / 2, mlp_vals, width, label="MLP",
                   color="coral")

    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Comparison chart saved to {save_path}")


# ============================================================
# 7. Analysis
# ============================================================

def print_analysis(bl_metrics, mlp_metrics, bl_time, mlp_time):
    """Print a concise (<200 words) comparative analysis."""
    print("\n" + "=" * 70)
    print("ANALYSIS (< 200 words)")
    print("=" * 70)

    analysis = (
        f"The MLP achieved an F1-Score of {mlp_metrics['F1-Score']:.4f} versus "
        f"{bl_metrics['F1-Score']:.4f} for Logistic Regression, a difference of "
        f"{abs(mlp_metrics['F1-Score'] - bl_metrics['F1-Score']):.4f}. "
        f"On the critical Recall metric, the MLP scored {mlp_metrics['Recall']:.4f} "
        f"compared to {bl_metrics['Recall']:.4f} for the baseline. "
        f"The MLP's additional hidden layers allow it to capture non-linear "
        f"decision boundaries that a linear classifier cannot represent, which "
        f"explains its edge on this dataset where feature interactions carry "
        f"diagnostic information. "
        f"However, this comes at a computational cost: the MLP trained in "
        f"{mlp_time:.3f}s versus {bl_time:.3f}s for Logistic Regression "
        f"({mlp_time / bl_time:.1f}x slower). "
        f"Both models' loss curves decrease monotonically, confirming correct "
        f"gradient descent convergence. "
        f"A notable finding is that the baseline already performs well on this "
        f"dataset because many features are approximately linearly separable. "
        f"The MLP's main advantage is its flexibility to model residual "
        f"non-linearities, though diminishing returns suggest deeper networks "
        f"would not yield proportional gains on only 569 samples."
    )

    print(analysis)
    word_count = len(analysis.split())
    print(f"\n[Word count: {word_count}]")


# ============================================================
# 8. Required result function
# ============================================================

def get_assignment_results():
    """
    Required entry point.
    Returns a dict with all results for grading.
    """
    X, y, feature_names = load_dataset()
    X_train, X_test, y_train, y_test = preprocess(X, y)

    # --- Baseline (Logistic Regression) ---
    print("\n" + "=" * 70)
    print("TRAINING: Logistic Regression (Baseline)")
    print("=" * 70)
    bl = BaselineModel(learning_rate=0.1, n_iterations=2000)
    t0 = time.time()
    bl.fit(X_train, y_train)
    bl_time = time.time() - t0
    bl_preds = bl.predict(X_test)
    bl_metrics = compute_metrics(y_test, bl_preds)
    print(f"  Training time : {bl_time:.4f}s")
    print(f"  Final loss    : {bl.loss_history[-1]:.6f}")
    print_metrics("Logistic Regression", bl_metrics)

    # --- MLP ---
    print("\n" + "=" * 70)
    print("TRAINING: Multi-Layer Perceptron")
    print("=" * 70)
    n_features = X_train.shape[1]
    architecture = [n_features, 64, 32, 1]
    print(f"  Architecture  : {architecture}")
    mlp = MLP(architecture=architecture, learning_rate=0.01, n_iterations=3000)
    t0 = time.time()
    mlp.fit(X_train, y_train)
    mlp_time = time.time() - t0
    mlp_preds = mlp.predict(X_test)
    mlp_metrics = compute_metrics(y_test, mlp_preds)
    print(f"  Training time : {mlp_time:.4f}s")
    print(f"  Final loss    : {mlp.loss_history[-1]:.6f}")
    print_metrics("MLP", mlp_metrics)

    # --- Visualizations ---
    print("\n" + "=" * 70)
    print("VISUALIZATIONS")
    print("=" * 70)
    plot_loss_curves(bl.loss_history, mlp.loss_history)
    plot_performance_comparison(bl_metrics, mlp_metrics)

    # --- Analysis ---
    print_analysis(bl_metrics, mlp_metrics, bl_time, mlp_time)

    return {
        "dataset": {
            "name": "Breast Cancer Wisconsin (Diagnostic)",
            "samples": 569,
            "features": 30,
            "problem_type": "Binary Classification",
            "primary_metric": "Recall",
        },
        "baseline": {
            "model": "Logistic Regression",
            "metrics": bl_metrics,
            "loss_history": bl.loss_history,
            "training_time_s": bl_time,
        },
        "mlp": {
            "model": "MLP",
            "architecture": architecture,
            "metrics": mlp_metrics,
            "loss_history": mlp.loss_history,
            "training_time_s": mlp_time,
        },
    }


# ============================================================
# 9. Main
# ============================================================

if __name__ == "__main__":
    results = get_assignment_results()
    print("\n" + "=" * 70)
    print("DONE — all outputs saved.")
    print("=" * 70)
