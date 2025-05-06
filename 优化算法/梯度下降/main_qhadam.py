from optimizer import QHAdamOptimizer
from torch_optimizer import QHAdam
from common import OptimizerComparator

if __name__ == '__main__':
    kwargs = {'lr': 10}
    configs = [{'type': 'system', 'class': QHAdam, 'kwargs': kwargs},
               {'type': 'custom', 'class': QHAdamOptimizer, 'kwargs': kwargs}]
    comparator = OptimizerComparator(configs=configs)
    comparator.compare()
