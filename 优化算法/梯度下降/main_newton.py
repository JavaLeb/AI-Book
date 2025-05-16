import torch
from optimizer import NewtonOptimizer
from common import OptimizerComparator

if __name__ == "__main__":
    kwargs1 = {'lr': 0.1,'max_iter':2000}
    kwargs2 = {'lr': 0.1}
    configs = [{'type': 'system', 'class': torch.optim.LBFGS, 'kwargs': kwargs1},
               {'type': 'custom', 'class': NewtonOptimizer, 'kwargs': kwargs2}]
    comparator = OptimizerComparator(configs=configs)
    comparator.compare()