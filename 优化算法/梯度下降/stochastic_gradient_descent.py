import torch


class StochasticGradientDescent:
    def __init__(self, params, lr=0.01, tol=1e-6):
        self.lr = lr
        self.tol = tol
        self.params = params

    def step(self):
        with torch.no_grad():
            for param in self.params:
                if param.grad is not None:
                    param.data -= self.lr * param.grad.data  # 自定义更新规则
                    param.grad.zero_()
