import torch
from optimizer import ASGDOptimizer, SystemOptimizer
from common import LROptimizerComparator

if __name__ == '__main__':
    lr = 0.01
    kwargs1 = {'lr': lr, 't0': 500, 'params': [torch.tensor([10.0, 10.0], requires_grad=True)]}
    kwargs2 = {'lr': lr, 't0': 500, 'params': torch.tensor([10.0, 10.0], requires_grad=True)}

    sys_optimizer = SystemOptimizer(optimizer=torch.optim.ASGD(**kwargs1))
    user_optimizer = ASGDOptimizer(**kwargs2)

    comparator = LROptimizerComparator(optimizers=[sys_optimizer, user_optimizer], batch_size=2)
    comparator.compare()
