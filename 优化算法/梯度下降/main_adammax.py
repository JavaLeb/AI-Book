import torch
from optimizer import AdamMaxOptimizer
from common import OptimizerComparator

if __name__ == '__main__':
    kwargs = {'lr': 1}
    configs = [{'type': 'system', 'class': torch.optim.Adamax, 'kwargs': kwargs},
               {'type': 'custom', 'class': AdamMaxOptimizer, 'kwargs': kwargs}]
    comparator = OptimizerComparator(configs=configs)
    comparator.compare()
