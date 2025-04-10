import torch


def generate_data():
    torch.manual_seed(42)
    n_samples = 10
    x = torch.linspace(0, 10, n_samples)  # shape: (10,).
    true_w, true_b = 2.0, 1.0
    y = true_w * x + true_b + torch.randn(x.size()) * 1.5  # shape: (10,)
    X = x.reshape(-1, 1)    # shape: (10,1)
    y = y.reshape(-1, 1)    # shape: (10,1)
    print('X=', X, 'y=', y)

    return X, y


