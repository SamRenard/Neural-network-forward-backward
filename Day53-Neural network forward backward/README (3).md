# Day 53 - Neural Network Forward & Backward Pass (NumPy from scratch)

[![CI](https://github.com/<your-username>/<your-repo>/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/<your-repo>/actions)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![NumPy](https://img.shields.io/badge/numpy-only-013243)
![License](https://img.shields.io/badge/license-MIT-green)

Part of the **NIZAM AI - 150 Day AI Engineering Protocol** (Month 2: Mathematics & Deep Learning).

A fully connected neural network with one hidden layer, implemented with **NumPy only**: forward pass,
softmax cross-entropy loss, hand-derived backpropagation, and gradient descent. Every gradient is
verified against numerical finite differences.

## Goals

- Write the **forward pass** of a one-hidden-layer network from scratch.
- Understand perceptrons, layers and the activation functions ReLU, sigmoid and tanh.
- Derive and implement the **backward pass**, then prove it correct with gradient checking.

## Features

- `TwoLayerNet`: `input -> hidden (ReLU | sigmoid | tanh) -> output (softmax)`.
- Vectorised over mini-batches (no Python loops over samples).
- Numerically stable `sigmoid`, `softmax` and `log_softmax` (log-sum-exp trick).
- He / Xavier weight initialisation chosen by activation function.
- Input and label validation with clear error messages.
- Reproducible runs through explicit seeding.
- Gradient-checked backward pass for all three activations.
- Continuous integration via GitHub Actions.

## Project structure

```
.
├── nn_numpy/
│   ├── __init__.py
│   ├── activations.py             # relu, sigmoid, tanh, softmax + derivatives
│   └── network.py                 # TwoLayerNet: forward, loss, backward, fit
├── examples/
│   ├── forward_pass_walkthrough.py  # hand-checkable forward pass + MNIST shapes
│   └── train_spiral.py              # train on a non-linearly-separable dataset
├── tests/
│   └── test_network.py
├── notes/
│   └── day53_summary.md           # short concept notes (Azerbaijani)
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

## Quick start

```python
import numpy as np
from nn_numpy import TwoLayerNet

net = TwoLayerNet(n_in=784, n_hidden=16, n_out=10, activation="relu", seed=0)
print(net)
# TwoLayerNet(784 -> 16 [relu] -> 10 [softmax], 12730 parameters)

X = np.random.default_rng(0).random((5, 784))   # 5 flattened 28x28 "images"
logits = net.forward(X)                          # (5, 10)
probs = net.predict_proba(X)                     # rows sum to 1
```

### Training

```python
history = net.fit(X_train, y_train, epochs=1000, lr=0.5)   # full-batch gradient descent
print(net.accuracy(X_test, y_test))
```

or step by step:

```python
net.forward(X)                  # caches intermediates
grads = net.backward(y)         # {"W1", "b1", "W2", "b2"}
net.step(grads, lr=0.1)
```

### Examples

```bash
python examples/forward_pass_walkthrough.py
python examples/train_spiral.py
```

Forward-pass walkthrough (verifiable by hand):

```
x      = [2. 1.]
z1     = [2.5 0.5]  (x @ W1 + b1)
a1     = [2.5 0.5]  (ReLU)
z2     = [2.1 5.4]  (a1 @ W2 + b2, the logits)
probs  = [0.0356 0.9644]  (softmax, sums to 1)
```

Spiral training (3 classes, 300 points, 64 hidden units, `lr=0.5`, 3000 epochs):

```
TwoLayerNet(2 -> 64 [relu] -> 3 [softmax], 387 parameters)
Initial loss 1.0676 (ln 3 = 1.0986) | accuracy 30.7%
epoch    0 | loss 1.0676
epoch  500 | loss 0.1168
epoch 1000 | loss 0.0767
epoch 1500 | loss 0.0604
epoch 2000 | loss 0.0511
epoch 2500 | loss 0.0451
Final loss 0.0408 | accuracy 99.3%
```

## The math

**Forward pass**

```
z1 = X  @ W1 + b1
a1 = f(z1)
z2 = a1 @ W2 + b2          (logits)
p  = softmax(z2)
L  = -(1/N) * sum_i log p[i, y_i]
```

**Backward pass**

```
dz2 = (p - one_hot(y)) / N
dW2 = a1.T @ dz2            db2 = sum(dz2, axis=0)
da1 = dz2 @ W2.T
dz1 = da1 * f'(z1)
dW1 = X.T  @ dz1            db1 = sum(dz1, axis=0)
```

| Activation | `f(z)` | `f'(z)` |
|------------|--------|---------|
| ReLU | `max(0, z)` | `1 if z > 0 else 0` |
| Sigmoid | `1 / (1 + e^-z)` | `s(1 - s)` |
| tanh | `tanh(z)` | `1 - tanh^2(z)` |

## Testing

```bash
python -m pytest -v
```

| Test | What it verifies |
|------|------------------|
| Activation values | Known outputs for ReLU, sigmoid, tanh |
| Activation gradients | Derivatives match central finite differences |
| Sigmoid stability | No overflow at +/-1000 |
| Softmax | Rows sum to 1, finite for logits around 1000 |
| Hand-computed forward | `[2, 1] -> [2.1, 5.4]` exactly |
| MNIST shapes | `(32, 784) -> (32, 10)`, 12,730 parameters |
| Input validation | Bad shapes, labels and activation names raise errors |
| Gradient check | Analytic gradients of `W1, b1, W2, b2` match numerical ones for ReLU, sigmoid and tanh |
| XOR | Network reaches 100% accuracy |
| Initial loss | Close to `ln(num_classes)` for small random inputs |

## Key takeaways

- A layer is one matrix multiplication followed by a non-linearity; the forward pass is just composition.
- Softmax + cross-entropy yields the elegantly simple gradient `p - y`.
- Non-linear activations are what give depth its power.
- A correct-looking backward pass is not proof; gradient checking is.

Short concept notes (in Azerbaijani): [`notes/day53_summary.md`](notes/day53_summary.md).

## Using real MNIST

The examples use synthetic data so they run offline. To train on MNIST, flatten each 28x28 image to a
784-vector scaled to `[0, 1]`, keep the labels as integer arrays, and pass them to `net.fit(...)`.
Full-batch training is slow on 60,000 images; splitting into mini-batches is the natural next step.

## Limitations

- One hidden layer only; no mini-batching, momentum, regularisation or dropout.
- Full-batch gradient descent is not intended for large datasets.
- CPU only.

## Roadmap

- [ ] Mini-batch SGD and learning-rate schedules
- [ ] Arbitrary-depth `MLP` class
- [ ] L2 regularisation and dropout
- [ ] Adam optimiser
- [ ] MNIST training script with accuracy report

## References

- 3Blue1Brown - [Neural Networks, chapter 1: But what is a neural network?](https://www.3blue1brown.com/topics/neural-networks)
- Andrej Karpathy - [Neural Networks: Zero to Hero (makemore)](https://karpathy.ai/zero-to-hero.html)
- [MNIST database of handwritten digits](http://yann.lecun.com/exdb/mnist/)
- [NumPy documentation](https://numpy.org/doc/)

## License

Released under the [MIT License](LICENSE).
