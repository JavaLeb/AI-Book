import torch
from common import VMOptimizerComparator
from optimizer import GDOptimizer, SystemGD, SystemOptimizer

if __name__ == '__main__':
    kwargs1 = {'lr': 0.01, 'params': [torch.tensor([-10.0, 10.0], requires_grad=True)]}
    kwargs2 = {'lr': 0.01, 'params': torch.tensor([10.0, 10.0], requires_grad=True)}
    sys_optimizer = SystemOptimizer(optimizer=SystemGD(**kwargs1))
    user_optimizer = GDOptimizer(**kwargs2)
    comparator = VMOptimizerComparator(optimizers=[sys_optimizer, user_optimizer])
    comparator.compare()
