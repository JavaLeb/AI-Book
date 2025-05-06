from torch_optimizer import LARS
from common import OptimizerComparator
from optimizer import LARSOptimizer

if __name__ == '__main__':
    lr = 0.1
    kwargs1 = {'lr': lr, 'momentum': 0.9, 'weight_decay': 0.01, 'trust_coefficient': 0.01}
    kwargs2 = {'lr': lr, 'momentum': 0.9, 'weight_decay': 0.01, 'lars_c': 0.01}
    configs = [{'type': 'system', 'class': LARS, 'kwargs': kwargs1},
               {'type': 'custom', 'class': LARSOptimizer, 'kwargs': kwargs2}]
    comparator = OptimizerComparator(configs=configs)
    comparator.compare()
