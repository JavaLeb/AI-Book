from optimizer import AggMoOptimizer
from torch_optimizer import AggMo
from common import OptimizerComparator

if __name__ == '__main__':
    kwargs = {'lr': 0.5}
    configs = [{'type': 'system', 'class': AggMo, 'kwargs': kwargs},
               {'type': 'custom', 'class': AggMoOptimizer, 'kwargs': kwargs}]
    comparator = OptimizerComparator(configs=configs)
    comparator.compare()
