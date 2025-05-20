import torch
from optimizer import LBFGSOptimizer, SystemOptimizer
from common import VMOptimizerComparator
from torch.optim import LBFGS

if __name__ == '__main__':
    lr = 0.1
    kwargs1 = {'lr': lr, 'max_iter': 2000, 'history_size': 5,
               'params': [torch.tensor([-10.0, 10.0], requires_grad=True)]}
    kwargs2 = {'lr': lr, 'history_size': 5, 'params': torch.tensor([10.0, 10.0], requires_grad=True)}
    sys_optimizer = SystemOptimizer(optimizer=LBFGS(**kwargs1))
    user_optimizer = LBFGSOptimizer(**kwargs2)
    comparator = VMOptimizerComparator(optimizers=[sys_optimizer, user_optimizer])
    comparator.compare()
