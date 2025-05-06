from optimizer import AdaBoundOptimizer, AMSBoundOptimizer
from common import OptimizerComparator
from torch_optimizer import AdaBound

if __name__ == '__main__':
    # AdaBound.
    kwargs = {'lr': 5}
    configs = [{'type': 'system', 'class': AdaBound, 'kwargs': kwargs},
               {'type': 'custom', 'class': AdaBoundOptimizer, 'kwargs': kwargs}]
    comparator = OptimizerComparator(configs=configs)
    comparator.compare()

    # AMSBound.
    kwargs1 = {'lr': 5, 'amsbound': True}
    kwargs2 = {'lr': 5}
    configs = [{'type': 'system', 'class': AdaBound, 'kwargs': kwargs1},
               {'type': 'custom', 'class': AMSBoundOptimizer, 'kwargs': kwargs2}]
    comparator = OptimizerComparator(configs=configs)
    comparator.compare()
