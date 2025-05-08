import torch
from optimizer import LionOptimizer
from common import OptimizerComparator
from pytorch_optimizer import Lion

if __name__ == '__main__':
    lr = 0.03
    kwargs1 = {'lr': lr, 'r': 0.0, 'betas': (0.9, 0.99), 'weight_decay': 0.1}
    kwargs2 = {'lr': lr, 'weight_decay': 0.1}
    configs = [{'type': 'system', 'class': Lion, 'kwargs': kwargs1},
               {'type': 'custom', 'class': LionOptimizer, 'kwargs': kwargs2}]
    comparator = OptimizerComparator(configs=configs)
    comparator.compare()
