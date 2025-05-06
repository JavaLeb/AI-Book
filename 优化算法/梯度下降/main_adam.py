import torch
from optimizer import AdamOptimizer
from common import OptimizerComparator

if __name__ == '__main__':
    kwargs = {'lr': 5}
    configs = [{'type': 'system', 'class': torch.optim.Adam, 'kwargs': kwargs},
               {'type': 'custom', 'class': AdamOptimizer, 'kwargs': kwargs}]
    comparator = OptimizerComparator(configs=configs)
    comparator.compare()
