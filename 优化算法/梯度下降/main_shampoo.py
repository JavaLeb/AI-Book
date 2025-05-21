import torch
from common import VirtualModel
from common import SquareFunction
from optimizer import ShampooOptimizer, SystemOptimizer
from common import VMOptimizerComparator
from torch_optimizer import Shampoo

if __name__ == '__main__':
    # 矩阵形式, shape=(1,2).
    lr = 5
    kwargs1 = {'lr': lr, 'params': [torch.tensor([[-10.0, 10.0]], requires_grad=True)]}
    kwargs2 = {'lr': lr, 'version': 0, 'params': torch.tensor([[10.0, 10.0]], requires_grad=True)}
    kwargs3 = {'lr': lr, 'version': 1, 'params': torch.tensor([[10.0, 10.0]], requires_grad=True)}
    sys_optimizer = SystemOptimizer(optimizer=Shampoo(**kwargs1))
    user_optimizer2 = ShampooOptimizer(**kwargs2)
    user_optimizer3 = ShampooOptimizer(**kwargs3)
    comparator = VMOptimizerComparator(optimizers=[sys_optimizer, user_optimizer2, user_optimizer3])
    comparator.compare()

    # 矩阵形式 shape=(2,3).
    criterion = SquareFunction(torch.tensor([[1.0, 3.0, 5.0],
                                             [2.0, 4.0, 6.0]]))
    for i in range(2):
        if i == 0:
            params = torch.tensor([[-10.0, -10.0, -10.0],
                                   [-10.0, -10.0, -10.0]], requires_grad=True)
            optimizer = SystemOptimizer(Shampoo(params=[params], lr=lr))
        else:
            params = torch.tensor([[10.0, 10.0, 10.0],
                                   [10.0, 10.0, 10.0]], requires_grad=True)
            optimizer = ShampooOptimizer(params=params, lr=lr)
        model = VirtualModel(
            criterion=criterion,
            optimizer=optimizer
        )
        model.fit()
