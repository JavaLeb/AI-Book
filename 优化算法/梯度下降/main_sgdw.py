from optimizer import  SGDOptimizer, SGDWOptimizer
from common import OptimizerComparator


if __name__ == '__main__':
    kwargs1 = {'lr': 0.01, 'gama': 2.0}
    kwargs2 = {'lr': 0.01, 'momentum': 0.0, 'gama': 2.0}
    configs = [
        {'type': 'custom', 'class': SGDOptimizer, 'kwargs': kwargs1},
               {'type': 'custom', 'class': SGDWOptimizer, 'kwargs': kwargs2}]
    comparator = OptimizerComparator(configs=configs)
    comparator.compare(is_model=True)
