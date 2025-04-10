import torch
from common import VirtualModel
from common import SquareFunction
from gradient_descent import GradientDescent

if __name__ == '__main__':
    initial_param = torch.tensor([10.0, 10.0], requires_grad=True)
    model = VirtualModel(
        loss=SquareFunction(torch.tensor([1.0, 2.0])),
        optimizer=GradientDescent(params=initial_param, lr=0.1)
    )
    model.fit()
