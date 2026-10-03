"""
Day 52 — Validation: compare our from-scratch Value engine against
real PyTorch autograd on the same expressions.
NIZAM AI · 150-Day AI Engineering Protocol
"""

import torch
from micrograd import Value


def run_case(label: str, our_fn, torch_fn) -> bool:
    our_result, our_grads = our_fn()
    torch_result, torch_grads = torch_fn()

    # PyTorch defaults to float32, our engine uses Python float64, so
    # allow a slightly looser tolerance for float32-precision drift.
    ok = abs(our_result - torch_result) < 1e-4
    for name in our_grads:
        ok = ok and abs(our_grads[name] - torch_grads[name]) < 1e-4

    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {label}")
    if not ok:
        print(f"  our:   result={our_result}, grads={our_grads}")
        print(f"  torch: result={torch_result}, grads={torch_grads}")
    return ok


def case_basic_expression():
    def ours():
        a = Value(2.0)
        b = Value(-3.0)
        c = Value(10.0)
        f = Value(-2.0)
        L = (a * b + c) * f
        L.backward()
        return L.data, {"a": a.grad, "b": b.grad, "c": c.grad, "f": f.grad}

    def torch_version():
        a = torch.tensor(2.0, requires_grad=True)
        b = torch.tensor(-3.0, requires_grad=True)
        c = torch.tensor(10.0, requires_grad=True)
        f = torch.tensor(-2.0, requires_grad=True)
        L = (a * b + c) * f
        L.backward()
        return L.item(), {"a": a.grad.item(), "b": b.grad.item(),
                           "c": c.grad.item(), "f": f.grad.item()}

    return run_case("basic expression (a*b+c)*f", ours, torch_version)


def case_tanh_neuron():
    def ours():
        x1 = Value(2.0)
        x2 = Value(0.0)
        w1 = Value(-3.0)
        w2 = Value(1.0)
        b = Value(6.8813735870195432)
        n = x1 * w1 + x2 * w2 + b
        out = n.tanh()
        out.backward()
        return out.data, {"x1": x1.grad, "x2": x2.grad, "w1": w1.grad, "w2": w2.grad}

    def torch_version():
        x1 = torch.tensor(2.0, requires_grad=True)
        x2 = torch.tensor(0.0, requires_grad=True)
        w1 = torch.tensor(-3.0, requires_grad=True)
        w2 = torch.tensor(1.0, requires_grad=True)
        b = torch.tensor(6.8813735870195432, requires_grad=True)
        n = x1 * w1 + x2 * w2 + b
        out = torch.tanh(n)
        out.backward()
        return out.item(), {"x1": x1.grad.item(), "x2": x2.grad.item(),
                             "w1": w1.grad.item(), "w2": w2.grad.item()}

    return run_case("tanh neuron", ours, torch_version)


def case_relu_and_division():
    def ours():
        a = Value(3.0)
        b = Value(-1.5)
        c = Value(4.0)
        out = (a.relu() + b.relu()) / c
        out.backward()
        return out.data, {"a": a.grad, "b": b.grad, "c": c.grad}

    def torch_version():
        a = torch.tensor(3.0, requires_grad=True)
        b = torch.tensor(-1.5, requires_grad=True)
        c = torch.tensor(4.0, requires_grad=True)
        out = (torch.relu(a) + torch.relu(b)) / c
        out.backward()
        return out.item(), {"a": a.grad.item(), "b": b.grad.item(), "c": c.grad.item()}

    return run_case("relu + division (tests __truediv__ and relu's zero-gradient branch)",
                     ours, torch_version)


def case_value_reused_multiple_times():
    """The classic gotcha: a Value used more than once in the graph must
    accumulate gradients from EVERY path, not just overwrite them."""
    def ours():
        x = Value(3.0)
        y = x + x  # x is used twice
        z = y * x  # x is used a third time
        z.backward()
        return z.data, {"x": x.grad}

    def torch_version():
        x = torch.tensor(3.0, requires_grad=True)
        y = x + x
        z = y * x
        z.backward()
        return z.item(), {"x": x.grad.item()}

    return run_case("value reused multiple times (gradient accumulation)",
                     ours, torch_version)


def case_power_and_exp():
    def ours():
        x = Value(2.0)
        out = (x ** 3) + x.exp()
        out.backward()
        return out.data, {"x": x.grad}

    def torch_version():
        x = torch.tensor(2.0, requires_grad=True)
        out = (x ** 3) + torch.exp(x)
        out.backward()
        return out.item(), {"x": x.grad.item()}

    return run_case("power rule + exp", ours, torch_version)


if __name__ == "__main__":
    results = [
        case_basic_expression(),
        case_tanh_neuron(),
        case_relu_and_division(),
        case_value_reused_multiple_times(),
        case_power_and_exp(),
    ]
    print(f"\n{sum(results)}/{len(results)} checks passed.")
    if not all(results):
        raise SystemExit(1)
