import torch_optimizer
from optimizer import YogiOptimizer
from common import OptimizerComparator

if __name__ == '__main__':
    lr = 0.1
    configs = [{'type': 'system', 'class': torch_optimizer.Yogi, 'kwargs': {'lr': lr}},
               {'type': 'custom', 'class': YogiOptimizer, 'kwargs': {'lr': lr}}]
    comparator = OptimizerComparator(configs=configs)
    comparator.compare()
