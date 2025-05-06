import torch
from optimizer import NewtonOptimizer
from common import OptimizerComparator

if __name__ == "__main__":
    kwargs = {'lr': 0.1}
    configs = [{'type': 'system', 'class': torch.optim.LBFGS, 'kwargs': kwargs},
               {'type': 'custom', 'class': NewtonOptimizer, 'kwargs': kwargs}]
    comparator = OptimizerComparator(configs=configs)
    comparator.compare()