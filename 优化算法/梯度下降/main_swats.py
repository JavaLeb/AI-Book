from optimizer import SWATSOptimizer
from torch_optimizer import SWATS
from common import OptimizerComparator

if __name__ == '__main__':
    lr = 5
    configs = [{'type': 'system', 'class': SWATS, 'kwargs': {'lr': lr}},
               {'type': 'custom', 'class': SWATSOptimizer, 'kwargs': {'lr': lr}}]
    comparator = OptimizerComparator(configs=configs)
    comparator.compare()
