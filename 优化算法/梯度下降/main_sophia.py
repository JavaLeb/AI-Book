import torch
from optimizer import SophiaOptimizer
from common import OptimizerComparator
from pytorch_optimizer import SophiaH

if __name__ == '__main__':
    lr = 2000
    configs = [{'type': 'system', 'class': SophiaH, 'kwargs': {'lr': lr}},
               {'type': 'custom', 'class': SophiaOptimizer, 'kwargs': {'lr': lr}}]
    comparator = OptimizerComparator(configs=configs)
    comparator.compare()
