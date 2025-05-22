import torch
from common import VirtualModel
from common import SquareFunction
from optimizer import AdafactorOptimizer, SystemOptimizer
from common import VMOptimizerComparator

if __name__ == '__main__':
    # 向量形式.
    lr = 5
    kwargs1 = {'lr': lr, 'params': [torch.tensor([-10.0, 10.0], requires_grad=True)]}
    kwargs2 = {'lr': lr, 'params': torch.tensor([10.0, 10.0], requires_grad=True)}
    sys_optimizer = SystemOptimizer(optimizer=torch.optim.Adafactor(**kwargs1))
    user_optimizer = AdafactorOptimizer(**kwargs2)
    comparator = VMOptimizerComparator(optimizers=[sys_optimizer, user_optimizer])
    comparator.compare()

    # 矩阵形式.
    lr = 5
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
