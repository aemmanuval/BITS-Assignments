"""
Deep Neural Networks — Programming Assignment 1
=================================================
Binary classification on the Online Shoppers Purchasing Intention dataset.

Models implemented from scratch (NumPy only):
  1. Logistic Regression  (baseline)
  2. Multi-Layer Perceptron (MLP)

Allowed sklearn usage: train_test_split, StandardScaler, LabelEncoder
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time

np.random.seed(42)

# ============================================================
# 1. Dataset Selection & Loading
# ============================================================

def load_dataset():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv"
    df = pd.read_csv(url)

    print("=" * 70)
    print("DATASET: Online Shoppers Purchasing Intention")
    print("=" * 70)
    print(f"Samples        : {df.shape[0]}")
    print(f"Features       : {df.shape[1] - 1}")
    print(f"Target         : Revenue (True/False)")
    print(f"Problem type   : Binary Classification")
    print(f"Class balance  : {dict(df['Revenue'].value_counts())}")
    print()

    return df


# ============================================================
# 2. Data Preprocessing
# ============================================================

def preprocess(df, test_size=0.2):
    le_month = LabelEncoder()
    df["Month"] = le_month.fit_transform(df["Month"])

    le_visitor = LabelEncoder()
    df["VisitorType"] = le_visitor.fit_transform(df["VisitorType"])

    df["Weekend"] = df["Weekend"].astype(int)
    df["Revenue"] = df["Revenue"].astype(int)

    X = df.drop("Revenue", axis=1).values.astype(float)
    y = df["Revenue"].values.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"Split ratio    : {1 - test_size:.0%} train / {test_size:.0%} test")
    print(f"Train samples  : {X_train.shape[0]}")
    print(f"Test  samples  : {X_test.shape[0]}")
    print(f"Missing values : {np.isnan(X_train).sum()}")
    print(f"Scaling        : StandardScaler")
    print()

    return X_train, X_test, y_train, y_test


# ============================================================
# 3. Baseline Model — Logistic Regression
# ============================================================

class BaselineModel:
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
            y_pred = self._sigmoid(X @ self.weights + self.bias)

            eps = 1e-15
            loss = -np.mean(
                y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps)
            )
            self.loss_history.append(loss)

            error = y_pred - y
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
# 4. Multi-Layer Perceptron
# ============================================================

class MLP:
    def __init__(self, architecture, learning_rate=0.01, n_iterations=1000):
        self.architecture = architecture
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.loss_history = []
        self.params = {}
        self.initialize_parameters()

    def initialize_parameters(self):
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
        y_col = y.reshape(-1, 1)
        A_out = cache[f"A{n_layers}"]

        dZ = A_out - y_col

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
# 5. Evaluation
# ============================================================

def compute_metrics(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    accuracy = float((tp + tn) / (tp + tn + fp + fn))
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-Score": f1}


# ============================================================
# 6. Visualizations
# ============================================================

def plot_loss_curves(bl_loss, mlp_loss, save_path="loss_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(bl_loss, color="steelblue")
    axes[0].set_title("Logistic Regression — Training Loss")
    axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("BCE Loss")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(mlp_loss, color="coral")
    axes[1].set_title("MLP — Training Loss")
    axes[1].set_xlabel("Iteration"); axes[1].set_ylabel("BCE Loss")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"  Loss curves saved to {save_path}")


def plot_performance_comparison(bl_m, mlp_m, save_path="performance_comparison.png"):
    labels = list(bl_m.keys())
    bl_v = [bl_m[k] for k in labels]
    mlp_v = [mlp_m[k] for k in labels]
    x = np.arange(len(labels)); w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, bl_v, w, label="Logistic Regression", color="steelblue")
    b2 = ax.bar(x + w/2, mlp_v, w, label="MLP", color="coral")
    ax.set_ylabel("Score"); ax.set_title("Model Performance Comparison")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.05)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for bar in b1:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{bar.get_height():.3f}", ha="center", fontsize=8)
    for bar in b2:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{bar.get_height():.3f}", ha="center", fontsize=8)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"  Comparison chart saved to {save_path}")


# ============================================================
# 7. Required result function
# ============================================================

def get_assignment_results():
    df = load_dataset()
    X_train, X_test, y_train, y_test = preprocess(df)

    print("\nTraining Logistic Regression...")
    bl = BaselineModel(learning_rate=0.1, n_iterations=2000)
    t0 = time.time()
    bl.fit(X_train, y_train)
    bl_time = time.time() - t0
    bl_preds = bl.predict(X_test)
    bl_m = compute_metrics(y_test, bl_preds)
    print(f"  Time: {bl_time:.4f}s | Final loss: {bl.loss_history[-1]:.6f}")
    for k, v in bl_m.items():
        print(f"  {k}: {v:.4f}")

    print("\nTraining MLP...")
    arch = [X_train.shape[1], 64, 32, 1]
    mlp = MLP(architecture=arch, learning_rate=0.01, n_iterations=3000)
    t0 = time.time()
    mlp.fit(X_train, y_train)
    mlp_time = time.time() - t0
    mlp_preds = mlp.predict(X_test)
    mlp_m = compute_metrics(y_test, mlp_preds)
    print(f"  Time: {mlp_time:.4f}s | Final loss: {mlp.loss_history[-1]:.6f}")
    for k, v in mlp_m.items():
        print(f"  {k}: {v:.4f}")

    plot_loss_curves(bl.loss_history, mlp.loss_history)
    plot_performance_comparison(bl_m, mlp_m)

    return {
        "dataset": {
            "name": "Online Shoppers Purchasing Intention",
            "samples": 12330,
            "features": 17,
            "problem_type": "Binary Classification",
            "primary_metric": "F1-Score",
        },
        "baseline": {
            "model": "Logistic Regression",
            "metrics": bl_m,
            "loss_history": bl.loss_history,
            "training_time_s": bl_time,
        },
        "mlp": {
            "model": "MLP",
            "architecture": arch,
            "metrics": mlp_m,
            "loss_history": mlp.loss_history,
            "training_time_s": mlp_time,
        },
    }


if __name__ == "__main__":
    results = get_assignment_results()
    print("\nDONE")
