import torch
from common import OptimizerComparator
from optimizer import NAGOptimizer,NesterovOptimizer



if __name__ == '__main__':
    configs = [{'type': 'system', 'class': torch.optim.SGD, 'kwargs': {'momentum': 0.9, 'lr': 0.1, 'nesterov': True}},
               {'type': 'custom', 'class': NAGOptimizer, 'kwargs': {'momentum': 0.9, 'lr': 0.1}},
               {'type': 'custom', 'class': NesterovOptimizer, 'kwargs': {'momentum': 0.9, 'lr': 0.1}}
               ]
    comparator = OptimizerComparator(configs=configs)
    comparator.compare()
