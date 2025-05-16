import torch
from optimizer import LBFGSOptimizer
from common import OptimizerComparator
from torch.optim import LBFGS

if __name__ == '__main__':
    lr = 0.01
    kwargs1 = {'lr': lr, 'max_iter': 2000, 'params': [torch.tensor([-10.0, 10.0], requires_grad=True)]}
    kwargs2 = {'lr': lr, 'params': torch.tensor([10.0, 10.0], requires_grad=True)}
    configs = [
        {'type': 'system', 'class': LBFGS, 'kwargs': kwargs1},
        {'type': 'custom', 'class': LBFGSOptimizer, 'kwargs': kwargs2}
    ]
    comparator = OptimizerComparator(configs=configs)
    comparator.compare()
