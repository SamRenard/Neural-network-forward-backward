"""
Day 57 — Full PyTorch Training Loop: MLP + Overfitting + Regularization
NIZAM AI · 150-Day AI Engineering Protocol

Goal: a complete, from-scratch PyTorch training loop for a digit
classifier, with a proper train/validation/test split, and a direct
A/B comparison showing overfitting happening (and being reduced by
dropout + weight decay).

NOTE ON DATA: the real MNIST dataset (28x28, 60k images) requires a
network download, which isn't available in this sandboxed environment
(outbound requests to the MNIST mirrors are blocked here). This script
uses scikit-learn's built-in `digits` dataset instead -- 1797 images of
handwritten digits 0-9, 8x8 pixels each -- which is MNIST's smaller
cousin and ships locally with scikit-learn, so it needs no download.
Every concept (train/val/test split, a PyTorch training loop, dropout,
weight decay, overfitting) is identical; only the data source differs.
To run this against real MNIST instead, see the "Using real MNIST"
section in the README -- it's a four-line swap.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def load_data(train_size: int = 150):
    """
    Loads sklearn's digits dataset and splits it into train/val/test.

    train_size is deliberately kept SMALL (default 150, vs. the ~1250
    that a 70% split would give) on purpose: overfitting is most
    visible and most pedagogically honest when the model has far more
    capacity than the training set has examples -- exactly the
    situation that makes regularization matter in practice. Validation
    and test still get a generous, representative held-out pool.
    """
    digits = load_digits()
    X = digits.data.astype(np.float32) / 16.0  # pixel values are 0-16, normalize to [0,1]
    y = digits.target.astype(np.int64)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=train_size, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    to_tensor = lambda a: torch.from_numpy(a)
    return (
        to_tensor(X_train), to_tensor(y_train),
        to_tensor(X_val), to_tensor(y_val),
        to_tensor(X_test), to_tensor(y_test),
    )


class MLP(nn.Module):
    """A simple 2-hidden-layer MLP, with optional dropout."""

    def __init__(self, in_features=64, hidden=128, n_classes=10, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
        return (preds == y).float().mean().item()


def train(model, X_train, y_train, X_val, y_val, epochs, lr, weight_decay, label=""):
    """
    A complete training loop: forward pass, loss, backward pass,
    optimizer step -- repeated for `epochs`, tracking train/val
    accuracy every epoch so we can see overfitting as it happens
    (train accuracy climbing while val accuracy stalls or drops).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    history = {"train_acc": [], "val_acc": [], "train_loss": []}

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()

        train_acc = accuracy(model, X_train, y_train)
        val_acc = accuracy(model, X_val, y_val)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(loss.item())

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"[{label}] epoch {epoch+1:4d}/{epochs}  "
                  f"loss={loss.item():.4f}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

    return history


if __name__ == "__main__":
    torch.manual_seed(0)
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    print(f"train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

    EPOCHS = 300

    print("\n" + "=" * 70)
    print("Model A: no regularization -- a big MLP, prone to overfitting")
    print("=" * 70)
    model_overfit = MLP(hidden=256, dropout=0.0)
    hist_overfit = train(
        model_overfit, X_train, y_train, X_val, y_val,
        epochs=EPOCHS, lr=1e-3, weight_decay=0.0, label="overfit",
    )

    print("\n" + "=" * 70)
    print("Model B: dropout + weight decay -- regularized")
    print("=" * 70)
    model_regularized = MLP(hidden=256, dropout=0.3)
    hist_regularized = train(
        model_regularized, X_train, y_train, X_val, y_val,
        epochs=EPOCHS, lr=1e-3, weight_decay=1e-3, label="regularized",
    )

    print("\n" + "=" * 70)
    print("Final comparison (held-out TEST set, touched only now)")
    print("=" * 70)
    train_test_gap_overfit = hist_overfit["train_acc"][-1] - accuracy(model_overfit, X_test, y_test)
    train_test_gap_reg = hist_regularized["train_acc"][-1] - accuracy(model_regularized, X_test, y_test)

    print(f"Model A (no regularization): train_acc={hist_overfit['train_acc'][-1]:.4f}  "
          f"test_acc={accuracy(model_overfit, X_test, y_test):.4f}  "
          f"gap={train_test_gap_overfit:.4f}")
    print(f"Model B (dropout+decay):     train_acc={hist_regularized['train_acc'][-1]:.4f}  "
          f"test_acc={accuracy(model_regularized, X_test, y_test):.4f}  "
          f"gap={train_test_gap_reg:.4f}")
    print(f"\n-> a smaller train/test gap means less overfitting: "
          f"Model A gap={train_test_gap_overfit:.4f} vs Model B gap={train_test_gap_reg:.4f}")
