"""
Day 52 — Autograd from Scratch: a minimal micrograd-style engine
NIZAM AI · 150-Day AI Engineering Protocol

Goal: rewrite Karpathy's `Value` class from understanding, not from
watching the video with the code open -- the point of today's practice
is to reconstruct backpropagation from first principles.

Core intuition: every arithmetic operation (+, *, **, tanh, exp, ...)
on a Value doesn't just compute a number -- it silently builds a node
in a computational graph, remembering which Values it came from (its
"children") and HOW to turn an incoming gradient into a gradient for
each child (the "local derivative" rule for that specific operation).
Calling .backward() on the final output walks this graph in reverse
topological order and applies the chain rule at every node:
  d(output)/d(child) = d(output)/d(self) * d(self)/d(child)
That's it -- backpropagation is nothing more than the chain rule,
applied automatically across a graph instead of by hand.
"""

from __future__ import annotations
import math


class Value:
    """A scalar value that tracks the computational graph that produced it,
    so gradients can be computed automatically via backpropagation."""

    def __init__(self, data: float, _children=(), _op: str = "", label: str = ""):
        self.data = data
        self.grad = 0.0
        self.label = label
        # _backward is a closure: "given my own .grad, how do I push
        # gradient into each of my children?" Defaults to a no-op for
        # leaf nodes (values with no children, e.g. raw inputs/weights).
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # ---------------- arithmetic operators ----------------

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(out)/d(self) = 1, d(out)/d(other) = 1 -> gradient just passes through.
            # += because a Value can be used multiple times in a graph
            # (e.g. x + x); gradients from every use must accumulate.
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # d(out)/d(self) = other.data, d(out)/d(other) = self.data
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), "only supports int/float powers"
        out = Value(self.data ** exponent, (self,), f"**{exponent}")

        def _backward():
            # power rule: d(x^n)/dx = n * x^(n-1)
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            # d(tanh(x))/dx = 1 - tanh(x)^2
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward():
            # d(exp(x))/dx = exp(x) itself
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():
            # gradient flows through only where the neuron was active
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    # ---------------- convenience operators (built from the above) ----------------

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other if isinstance(other, Value) else -other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        return self * other ** -1

    def __rtruediv__(self, other):
        return other * self ** -1

    # ---------------- the core algorithm: backward() ----------------

    def backward(self):
        """
        Runs backpropagation from this Value: computes d(self)/d(v)
        for every Value v in the graph that fed into this one.

        The graph must be visited in REVERSE topological order --
        every node's own gradient has to be fully accumulated (from
        ALL paths that use it) before it pushes gradient further back
        to its own children. A plain DFS without this ordering would
        push partial/incorrect gradients through a node with multiple
        parents before all its parents had contributed.
        """
        topo_order = []
        visited = set()

        def build_topo(v: "Value"):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo_order.append(v)  # a node is appended only AFTER all its children

        build_topo(self)

        self.grad = 1.0  # d(self)/d(self) = 1, the seed for the chain rule
        for v in reversed(topo_order):
            v._backward()


if __name__ == "__main__":
    # --- Example 1: a tiny expression, gradients checked against hand calculus ---
    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    c = Value(10.0, label="c")
    e = a * b
    d = e + c
    f = Value(-2.0, label="f")
    L = d * f

    L.backward()

    print("Expression: L = (a*b + c) * f")
    print(f"a={a.data}, b={b.data}, c={c.data}, f={f.data}  ->  L={L.data}")
    print(f"dL/da = {a.grad}  (expected: b*f = {b.data * f.data})")
    print(f"dL/db = {b.grad}  (expected: a*f = {a.data * f.data})")
    print(f"dL/dc = {c.grad}  (expected: f   = {f.data})")
    print(f"dL/df = {f.grad}  (expected: a*b+c = {a.data * b.data + c.data})")

    # --- Example 2: a single "neuron" -- exactly what a real NN layer computes ---
    print("\n" + "=" * 60)
    print("A tiny neuron: tanh(w1*x1 + w2*x2 + b)")
    print("=" * 60)
    x1 = Value(2.0, label="x1")
    x2 = Value(0.0, label="x2")
    w1 = Value(-3.0, label="w1")
    w2 = Value(1.0, label="w2")
    bias = Value(6.8813735870195432, label="b")

    n = x1 * w1 + x2 * w2 + bias
    out = n.tanh()
    out.backward()

    print(f"output = {out.data:.6f}")
    print(f"d(out)/dw1 = {w1.grad:.6f}")
    print(f"d(out)/dw2 = {w2.grad:.6f}")
    print(f"d(out)/dx1 = {x1.grad:.6f}")
    print(f"d(out)/dx2 = {x2.grad:.6f}")
