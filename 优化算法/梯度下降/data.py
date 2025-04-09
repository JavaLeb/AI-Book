import torch


def generate_data():
    torch.manual_seed(42)
    x = torch.linspace(0, 10, 10)  # shape: (10,).
    true_w, true_b = 2.0, 1.0
    y = true_w * x + true_b + torch.randn(x.size()) * 1.5  # shape: (10,)
    X = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    return X, y
