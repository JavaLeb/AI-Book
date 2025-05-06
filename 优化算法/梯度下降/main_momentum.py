import torch
from common import OptimizerComparator
from optimizer import MomentumOptimizer

if __name__ == '__main__':
    kwargs = {'momentum': 0.9, 'lr': 0.01}
    configs = [{'type': 'system', 'class': torch.optim.SGD, 'kwargs': kwargs},
               {'type': 'custom', 'class': MomentumOptimizer, 'kwargs': kwargs}]
    comparator = OptimizerComparator(configs=configs)
    comparator.compare()
