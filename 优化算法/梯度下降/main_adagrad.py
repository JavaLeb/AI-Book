import torch
from optimizer import AdaGradOptimizer
from common import OptimizerComparator

if __name__ == '__main__':
    lr = 10
    configs = [{'type': 'system', 'class': torch.optim.Adagrad, 'kwargs': {'lr': lr}},
               {'type': 'custom', 'class': AdaGradOptimizer, 'kwargs': {'lr': lr}}]
    comparator = OptimizerComparator(configs=configs)
    comparator.compare()
