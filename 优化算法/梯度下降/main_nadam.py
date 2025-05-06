import torch
from optimizer import NadamOptimizer
from common import OptimizerComparator

if __name__ == '__main__':
    lr = 0.1
    configs = [{'type': 'system', 'class': torch.optim.NAdam, 'kwargs': {'lr': lr}},
               {'type': 'custom', 'class': NadamOptimizer, 'kwargs': {'lr': lr}}]
    comparator = OptimizerComparator(configs=configs)
    comparator.compare()
