# Day 52 — Autograd from Scratch (micrograd-style)

Part of the **NIZAM AI · 150-Day AI Engineering Protocol**
Month 2 · Mathematics & Deep Learning · Block Day 5/5

## 📌 Concept

PyTorch's "magic" is one idea: every arithmetic operation silently
builds a **computational graph** node, remembering which values it
came from and how to turn an incoming gradient into a gradient for
each of those values (the operation's local derivative rule). Calling
`.backward()` walks this graph in **reverse topological order** and
applies the **chain rule** at every node — that's the entirety of
backpropagation.

## 📂 Files

| File | Description |
|---|---|
| `micrograd.py` | `Value` class rewritten from understanding — `+`, `*`, `**`, `tanh`, `exp`, `relu`, division, and `backward()` with topological sort |
| `validate_with_pytorch.py` | Cross-checks every operation's output and gradients against real `torch.autograd` |

## ▶️ Usage

```bash
pip install torch
python3 micrograd.py              # run the classic Karpathy examples
python3 validate_with_pytorch.py  # validate against real PyTorch
```

## ✅ Test Results

```
5/5 checks passed
```

Covers: a basic multi-op expression, a tanh neuron (the canonical
micrograd example), relu + division, a value reused multiple times in
the graph (gradient accumulation — the classic bug source), and the
power rule combined with exp.

## 🔑 Key takeaway

Gradients must **accumulate** (`+=`), not overwrite, because a single
`Value` can appear in the graph more than once (e.g. `y = x + x`).
Overwriting instead of accumulating is the most common bug when
implementing this from scratch — `test_value_reused_multiple_times`
exists specifically to catch it.

## 📚 Sources

- Karpathy — "The spelled-out intro to neural networks and
  backpropagation: building micrograd"
- Karpathy — Zero to Hero playlist

---
*NIZAM AI Day 52/150*
