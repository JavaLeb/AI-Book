import torch
from common import OptimizerComparator
from optimizer import SGDOptimizer

if __name__ == '__main__':
    kwargs = {'lr': 0.01}
    configs = [{'type': 'system', 'class': torch.optim.SGD, 'kwargs': kwargs,
                'initial_param': torch.tensor([10.0, 10.0], requires_grad=True)},
               {'type': 'custom', 'class': SGDOptimizer, 'kwargs': kwargs,
                'initial_param': torch.tensor([10.0, 10.0], requires_grad=True)}]
    comparator = OptimizerComparator(configs=configs)
    comparator.compare(is_model=True)
