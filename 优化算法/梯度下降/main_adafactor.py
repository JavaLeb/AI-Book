import torch
from common import VirtualModel
from common import SquareFunction
from optimizer import AdafactorOptimizer, SystemOptimizer
from common import OptimizerComparator

if __name__ == '__main__':
    # 向量形式.
    lr = 0.1
    configs = [{'type': 'system', 'class': torch.optim.Adafactor, 'kwargs': {'lr': lr}},
               {'type': 'custom', 'class': AdafactorOptimizer, 'kwargs': {'lr': lr}}]
    comparator = OptimizerComparator(configs=configs)
    comparator.compare()

    # 矩阵形式.
    lr = 0.1
    criterion = SquareFunction(torch.tensor([[1.0, 2.0, 3.0],
                                             [4.0, 5.0, 6.0]]))
    for i in range(2):
        params = torch.tensor([[10.0, 10.0, 10.0],
                               [10.0, 10.0, 10.0]], requires_grad=True)
        optimizer = SystemOptimizer(torch.optim.Adafactor(params=[params], lr=lr)) if i == 0 else AdafactorOptimizer(
            params=params, lr=lr)
        model = VirtualModel(
            criterion=criterion,
            optimizer=optimizer
        )
        model.fit()
