import torch
from optimizer import ASGDOptimizer, SystemOptimizer
from common import VMOptimizerComparator

if __name__ == '__main__':
    lr = 0.01
    kwargs1 = {'lr': lr, 't0': 3, 'lambd': 0.0, 'params': [torch.tensor([-10.0, 10.0], requires_grad=True)]}
    kwargs2 = {'lr': lr, 't0': 3, 'params': torch.tensor([10.0, 10.0], requires_grad=True)}

    sys_optimizer = SystemOptimizer(optimizer=torch.optim.ASGD(**kwargs1))
    user_optimizer = ASGDOptimizer(**kwargs2)
    comparator = VMOptimizerComparator(optimizers=[sys_optimizer, user_optimizer])
    comparator.compare()
