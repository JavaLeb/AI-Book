import torch_optimizer
from optimizer import MADGRADOptimizer
from common import OptimizerComparator


if __name__ == '__main__':
    lr = 10
    configs = [{'type': 'system', 'class': torch_optimizer.MADGRAD, 'kwargs': {'lr': lr}},
               {'type': 'custom', 'class': MADGRADOptimizer, 'kwargs': {'lr': lr}}]
    comparator = OptimizerComparator(configs=configs)
    comparator.compare()