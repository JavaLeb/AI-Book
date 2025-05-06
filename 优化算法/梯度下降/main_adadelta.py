import torch
from optimizer import AdaDeltaOptimizer
from common import OptimizerComparator

if __name__ == '__main__':
    lr = 2000
    configs = [{'type': 'system', 'class': torch.optim.Adadelta, 'kwargs': {'lr': lr}},
               {'type': 'custom', 'class': AdaDeltaOptimizer, 'kwargs': {'lr': lr}}]
    comparator = OptimizerComparator(configs=configs)
    comparator.compare()
