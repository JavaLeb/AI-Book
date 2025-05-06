import torch
from optimizer import RpropOptimizer
from common import OptimizerComparator

if __name__ == '__main__':
    lr = 0.8
    configs = [{'type': 'system', 'class': torch.optim.Rprop, 'kwargs': {'lr': lr}},
               {'type': 'custom', 'class': RpropOptimizer, 'kwargs': {'lr': lr}}]
    comparator = OptimizerComparator(configs=configs)
    comparator.compare()
