import torch
from virtualmodel import VirtualModel
from loss import Loss
from gradient_descent import GradientDescent

if __name__ == '__main__':
    initial_param = torch.tensor([10.0, 10.0], requires_grad=True)
    model = VirtualModel(
        loss=Loss(),
        optimizer=GradientDescent(params=[initial_param], lr=0.1)
    )
    model.fit()
