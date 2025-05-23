import torch
from optimizer import LBFGSOptimizer, SystemOptimizer, BFGSOptimizer, NewtonOptimizer
from common import VMOptimizerComparator
from torch.optim import LBFGS

if __name__ == '__main__':
    lr = 0.1
    kwargs1 = {'lr': lr, 'max_iter': 2000, 'history_size': 5,
               'params': [torch.tensor([-10.0, 10.0], requires_grad=True)]}
    kwargs2 = {'lr': lr, 'history_size': 5, 'params': torch.tensor([10.0, 10.0], requires_grad=True)}
    kwargs3 = {'lr': lr, 'params': torch.tensor([-10.0, -10.0], requires_grad=True)}
    kwargs4 = {'lr': lr, 'params': torch.tensor([10.0, -10.0], requires_grad=True)}
    sys_optimizer = SystemOptimizer(optimizer=LBFGS(**kwargs1))
    user_op2 = LBFGSOptimizer(**kwargs2)
    user_op3 = NewtonOptimizer(**kwargs3)
    user_op4 = BFGSOptimizer(**kwargs4)
    comparator = VMOptimizerComparator(optimizers=[sys_optimizer, user_op2, user_op3, user_op4])
    comparator.compare()
