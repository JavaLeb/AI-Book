import torch
from optimizer import ASGDOptimizer
from common import OptimizerComparator

if __name__ == '__main__':
    kwargs1 = {'lr': 0.01, 't0': 3, 'lambd': 0.0}
    kwargs2 = {'lr': 0.01, 't0': 3}
    configs = [{'type': 'system', 'class': torch.optim.ASGD, 'kwargs': kwargs1,
                'initial_param': torch.tensor([10.0, 10.0], requires_grad=True)},
               {'type': 'custom', 'class': ASGDOptimizer, 'kwargs': kwargs2,
                'initial_param': torch.tensor([10.0, 10.0], requires_grad=True)}]
    comparator = OptimizerComparator(configs=configs)
    comparator.compare(is_model=True)
