# Day 54 - Backpropagation from Scratch (NumPy)

[![CI](https://github.com/<your-username>/<your-repo>/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/<your-repo>/actions)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![NumPy](https://img.shields.io/badge/numpy-only-013243)
![License](https://img.shields.io/badge/license-MIT-green)

Part of the **NIZAM AI - 150 Day AI Engineering Protocol** (Month 2: Mathematics & Deep Learning).

A multi-layer perceptron of **arbitrary depth** in pure NumPy with a **hand-derived backward pass**.
Every gradient is checked against numerical finite differences *and* against a deliberately naive,
loop-based implementation, so the vectorised code is proven correct, not just plausible.

This builds on Day 53 (forward pass for one hidden layer) by deriving backpropagation for any number of
layers, adding a second loss function, L2 regularisation and mini-batch SGD.

## Goals

- Derive backpropagation by hand and implement it.
- Generalise it from one hidden layer to any depth.
- Prove correctness with gradient checking and an independent reference implementation.

## Features

- `MLP([n_in, h1, ..., n_out])` with `relu`, `sigmoid` or `tanh` hidden layers.
- Losses: softmax cross-entropy (classification) and mean squared error (regression).
- Optional L2 weight decay, folded into both the loss and its gradient.
- Full-batch or shuffled mini-batch SGD with reproducible seeding.
- `check_gradients`: relative-error report per parameter.
- `naive_backward`: per-sample, loop-based reference backward pass.
- Numerically stable `sigmoid`, `softmax` and `log_softmax`.
- Input, label and argument validation with clear errors.
- Continuous integration via GitHub Actions.

## Project structure

```
.
├── nn_numpy/
│   ├── __init__.py
│   ├── activations.py      # relu, sigmoid, tanh, softmax + derivatives
│   ├── network.py          # MLP: forward, loss, backward, fit
│   ├── gradcheck.py        # numerical gradients and relative error
│   └── reference.py        # naive per-sample backward pass for verification
├── examples/
│   ├── backprop_step_by_step.py   # one sample, every gradient printed and verified
│   ├── train_spiral.py            # deeper net: gradient check + mini-batch training
│   └── train_digits.py            # 8x8 handwritten digits (bundled with scikit-learn)
├── tests/
│   └── test_backprop.py
├── notes/
│   └── day54_summary.md    # short concept notes (Azerbaijani)
├── .github/workflows/ci.yml
├── pytest.ini
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

scikit-learn is needed only for `examples/train_digits.py`; the library itself depends on NumPy alone.

## Quick start

```python
from nn_numpy import MLP, check_gradients

net = MLP([64, 64, 32, 10], activation="relu", l2=1e-4, seed=0)
print(net)
# MLP(64 -> 64 -> 32 -> 10, hidden=relu, loss=cross_entropy, 6570 parameters)

history = net.fit(X_train, y_train, epochs=60, lr=0.1, batch_size=32)
print(net.accuracy(X_test, y_test))
```

Manual training step:

```python
net.forward(X_batch)             # caches every intermediate value
grads = net.backward(y_batch)    # {"W1", "b1", "W2", "b2", ...}
net.step(grads, lr=0.1)
```

Always gradient-check a new implementation:

```python
print(check_gradients(net, X[:20], y[:20], eps=1e-6))
# {'W3': 2.3e-09, 'b3': 1.0e-10, 'W2': 3.3e-09, ...}   all should be below ~1e-6
```

## Examples and results

```bash
python examples/backprop_step_by_step.py
python examples/train_spiral.py
python examples/train_digits.py
```

**Step by step.** A 2 -> 3 -> 2 network on one sample. Every gradient is computed by hand in the script
and compared with `MLP.backward` (relative error `0`) and with finite differences (about `1e-11`).

**Spiral, 3 classes** (`2 -> 64 -> 32 -> 3`, ReLU, L2 = 1e-4, mini-batch 32, 300 epochs):

```
Gradient check (relative error, should be < 1e-6):
  W3: 2.34e-09   b3: 1.02e-10   W2: 3.25e-09   b2: 3.28e-10   W1: 4.75e-09   b1: 3.89e-10

Before training: loss 1.2308 | accuracy 28.3%
After training:  loss 0.0582 | accuracy 98.3%
```

**Handwritten digits** (`64 -> 64 -> 32 -> 10`, 80/20 split, 60 epochs):

```
Before training: test accuracy 9.2%
Train accuracy: 100.0%
Test accuracy:  98.3%
```

## The math

Layers are numbered `1..L`, with `A[0] = X`.

**Forward**

```
Z[i] = A[i-1] @ W[i] + b[i]
A[i] = f(Z[i])              for hidden layers
out  = Z[L]                 linear output
```

**Backward** (N = batch size)

```
dZ[L]   = (softmax(out) - one_hot(y)) / N      cross-entropy
        = 2 (out - y) / N                      mean squared error
dW[i]   = A[i-1].T @ dZ[i] + l2 * W[i]
db[i]   = sum(dZ[i], axis=0)
dA[i-1] = dZ[i] @ W[i].T
dZ[i-1] = dA[i-1] * f'(Z[i-1])
```

The cross-entropy gradient is so simple because the softmax Jacobian and the `log` derivative cancel,
leaving `p - y`.

## Testing

```bash
python -m pytest -v
```

| Test | What it verifies |
|------|------------------|
| Hand-computed forward | `[2, 1] -> [2.1, 5.4]` exactly |
| Gradient check, 3 activations x 2 losses | Analytic vs numerical gradients below `1e-6` |
| Gradient check, depths 1 to 4 hidden layers | The recursion generalises to any depth |
| Gradient check with L2 | The regulariser's gradient is correct |
| Vectorised vs naive backward | Matches the per-sample loop to `1e-10`, all 6 combinations |
| Sabotage test | A gradient that is 10% wrong is flagged by `relative_error` |
| XOR | Full-batch training reaches 100% accuracy |
| Mini-batch | Loss decreases and runs are reproducible with a fixed seed |
| MSE regression | Linear target is fitted to loss below `1e-4` |
| L2 | Weight decay produces smaller weights |
| Validation | Bad shapes, labels, losses and activations raise errors |

## Gradient-checking pitfalls

Two non-obvious failures came up while building the examples, and neither was a bug in the backward pass:

- **ReLU kink.** At `z = 0` the derivative is undefined. The spiral dataset starts each arm at exactly
  `(0, 0)`, so with zero-initialised biases `z = 0` exactly and the numerical and analytic gradients
  disagree (about `2e-2` error on the biases). Avoid such points when checking.
- **Step size.** If any pre-activation lies within `eps` of zero, the finite difference straddles the kink.
  With `eps = 1e-5` one unit sat at `|z| = 4.7e-6` and the bias error rose to `4e-4`; with `eps = 1e-6`
  it dropped to `4e-10`.

Smooth activations (`tanh`, `sigmoid`) do not have this issue.

## Key takeaways

- Backpropagation is the chain rule applied layer by layer, reusing values cached in the forward pass.
- One recursion, `dZ[i-1] = (dZ[i] @ W[i].T) * f'(Z[i-1])`, handles any depth.
- A backward pass that "looks right" is not evidence; gradient checking and a reference implementation are.
- When a gradient check fails, investigate the check itself (kinks, step size) before the model.

Short concept notes (in Azerbaijani): [`notes/day54_summary.md`](notes/day54_summary.md).

## Limitations

- Plain SGD only: no momentum, Adam or learning-rate schedules.
- No dropout or batch normalisation.
- Gradient checking is `O(parameters)` forward passes, so use it on small batches.
- CPU only.

## Roadmap

- [ ] Momentum and Adam optimisers
- [ ] Learning-rate schedules
- [ ] Dropout and batch normalisation (with their backward passes)
- [ ] Full 28x28 MNIST training script

## References

- 3Blue1Brown - [Neural Networks, chapters 3-4: Backpropagation](https://www.3blue1brown.com/topics/neural-networks)
- Andrej Karpathy - [Neural Networks: Zero to Hero (makemore)](https://karpathy.ai/zero-to-hero.html)
- [MNIST database of handwritten digits](http://yann.lecun.com/exdb/mnist/)
- [scikit-learn digits dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)

## License

Released under the [MIT License](LICENSE).
