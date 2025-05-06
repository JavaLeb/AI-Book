from torch_optimizer import Lookahead
from common import OptimizerComparator
from optimizer import LookaheadOptimizer

if __name__ == '__main__':
    lr = 0.1
    kwargs1 = {'lr': lr, 'weight_decay': 0.01, 'debias': False}
    kwargs2 = {'lr': lr, 'weight_decay': 0.01}
    configs = [{'type': 'system', 'class': Lookahead, 'kwargs': kwargs1},
               {'type': 'custom', 'class': LookaheadOptimizer, 'kwargs': kwargs2}]
    comparator = OptimizerComparator(configs=configs)
    comparator.compare()
