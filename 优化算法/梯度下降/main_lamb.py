from torch_optimizer import Lamb
from common import OptimizerComparator
from optimizer import LambOptimizer

if __name__ == '__main__':
    lr = 0.1
    kwargs1 = {'lr': lr, 'weight_decay': 0.01, 'debias': False}
    kwargs2 = {'lr': lr, 'weight_decay': 0.01}
    configs = [{'type': 'system', 'class': Lamb, 'kwargs': kwargs1},
               {'type': 'custom', 'class': LambOptimizer, 'kwargs': kwargs2}]
    comparator = OptimizerComparator(configs=configs)
    comparator.compare()
