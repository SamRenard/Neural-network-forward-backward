"""
Day 57 — Automated correctness checks for mlp_train_loop.py
NIZAM AI · 150-Day AI Engineering Protocol
"""

import torch
from mlp_train_loop import MLP, load_data, train, accuracy


def test_data_split_sizes_and_no_overlap():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(train_size=150)
    assert len(X_train) == 150
    assert len(X_val) > 0 and len(X_test) > 0
    # total should equal the full dataset size (1797 for sklearn digits)
    assert len(X_train) + len(X_val) + len(X_test) == 1797
    print("[PASS] train/val/test split sizes are correct and cover the full dataset")


def test_all_classes_present_in_each_split():
    """Stratified split should mean all 10 digit classes appear in train/val/test."""
    _, y_train, _, y_val, _, y_test = load_data(train_size=150)
    assert len(torch.unique(y_train)) == 10
    assert len(torch.unique(y_val)) == 10
    assert len(torch.unique(y_test)) == 10
    print("[PASS] all 10 digit classes present in train, val, and test splits")


def test_model_forward_pass_shape():
    model = MLP(in_features=64, hidden=32, n_classes=10)
    x = torch.randn(5, 64)
    out = model(x)
    assert out.shape == (5, 10)
    print("[PASS] MLP forward pass produces correct output shape")


def test_dropout_changes_output_in_train_mode():
    """With dropout active and train() mode, repeated forward passes on
    the same input should NOT be identical (neurons are randomly dropped)."""
    torch.manual_seed(1)
    model = MLP(hidden=64, dropout=0.5)
    model.train()
    x = torch.randn(1, 64)
    out1 = model(x)
    out2 = model(x)
    assert not torch.allclose(out1, out2)
    print("[PASS] dropout produces different outputs across forward passes in train mode")


def test_dropout_is_deterministic_in_eval_mode():
    """In eval() mode, dropout should be disabled -- identical input
    must give identical output every time."""
    torch.manual_seed(1)
    model = MLP(hidden=64, dropout=0.5)
    model.eval()
    x = torch.randn(1, 64)
    out1 = model(x)
    out2 = model(x)
    assert torch.allclose(out1, out2)
    print("[PASS] dropout is disabled in eval mode (deterministic output)")


def test_training_reduces_loss():
    torch.manual_seed(0)
    X_train, y_train, X_val, y_val, _, _ = load_data(train_size=150)
    model = MLP(hidden=64, dropout=0.0)
    hist = train(model, X_train, y_train, X_val, y_val,
                 epochs=100, lr=1e-3, weight_decay=0.0, label="test")
    assert hist["train_loss"][-1] < hist["train_loss"][0]
    assert hist["train_acc"][-1] > hist["train_acc"][0]
    print(f"[PASS] training reduces loss ({hist['train_loss'][0]:.3f} -> "
          f"{hist['train_loss'][-1]:.3f}) and increases train accuracy")


def test_regularization_reduces_train_val_gap():
    """
    The core claim of today's lesson: with a small training set and a
    high-capacity model, a regularized model (dropout + weight decay)
    should show a smaller gap between train and validation accuracy
    than an unregularized one, trained for the same number of epochs.
    """
    torch.manual_seed(0)
    X_train, y_train, X_val, y_val, _, _ = load_data(train_size=150)

    model_plain = MLP(hidden=256, dropout=0.0)
    hist_plain = train(model_plain, X_train, y_train, X_val, y_val,
                        epochs=200, lr=1e-3, weight_decay=0.0, label="plain")

    model_reg = MLP(hidden=256, dropout=0.3)
    hist_reg = train(model_reg, X_train, y_train, X_val, y_val,
                      epochs=200, lr=1e-3, weight_decay=1e-3, label="reg")

    gap_plain = hist_plain["train_acc"][-1] - hist_plain["val_acc"][-1]
    gap_reg = hist_reg["train_acc"][-1] - hist_reg["val_acc"][-1]

    assert gap_reg < gap_plain
    print(f"[PASS] regularized model has smaller train/val gap "
          f"({gap_reg:.4f}) than unregularized ({gap_plain:.4f})")


if __name__ == "__main__":
    tests = [
        test_data_split_sizes_and_no_overlap,
        test_all_classes_present_in_each_split,
        test_model_forward_pass_shape,
        test_dropout_changes_output_in_train_mode,
        test_dropout_is_deterministic_in_eval_mode,
        test_training_reduces_loss,
        test_regularization_reduces_train_val_gap,
    ]
    for t in tests:
        t()
    print(f"\n{len(tests)}/{len(tests)} checks passed.")
