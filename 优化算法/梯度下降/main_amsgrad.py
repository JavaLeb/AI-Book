import torch
from optimizer import AMSGradOptimizer
from common import OptimizerComparator

if __name__ == '__main__':
    lr = 0.9
    kwargs1 = {'lr': lr, 'amsgrad': True}
    kwargs2 = {'lr': lr, 'momentum': 0.9, 'decay_factor': 0.999}
    configs = [{'type': 'system', 'class': torch.optim.Adam, 'kwargs': kwargs1},
               {'type': 'custom', 'class': AMSGradOptimizer, 'kwargs': kwargs2}]
    comparator = OptimizerComparator(configs=configs)
    comparator.compare()
