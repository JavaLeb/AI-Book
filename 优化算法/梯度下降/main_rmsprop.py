import torch
from common import OptimizerComparator
from optimizer import RMSPropOptimizer

if __name__ == '__main__':
    kwargs = {'lr': 1.0}
    configs = [{'type': 'system', 'class': torch.optim.RMSprop, 'kwargs': kwargs},
               {'type': 'custom', 'class': RMSPropOptimizer, 'kwargs': kwargs}]
    comparator = OptimizerComparator(configs=configs)
    comparator.compare()
