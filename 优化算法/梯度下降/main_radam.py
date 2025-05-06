import torch
from optimizer import RAdamOptimizer
from common import OptimizerComparator

if __name__ == '__main__':
    kwargs = {'lr': 0.1}
    configs = [{'type': 'system', 'class': torch.optim.RAdam, 'kwargs': kwargs},
               {'type': 'custom', 'class': RAdamOptimizer, 'kwargs': kwargs}]
    comparator = OptimizerComparator(configs=configs)
    comparator.compare()
