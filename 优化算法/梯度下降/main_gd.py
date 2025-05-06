import torch
from common import OptimizerComparator
from optimizer import GDOptimizer

if __name__ == '__main__':
    kwargs = {'lr': 0.01}
    configs = [{'type': 'system', 'class': torch.optim.SGD, 'kwargs': kwargs},
               {'type': 'custom', 'class': GDOptimizer, 'kwargs': kwargs}]
    comparator = OptimizerComparator(configs=configs)
    comparator.compare()

    # 线性回归.
    kwargs = {'lr': 0.01}
    configs = [{'type': 'system', 'class': torch.optim.SGD, 'kwargs': kwargs,
                'initial_param': torch.tensor([10.0, 10.0], requires_grad=True)},
               {'type': 'custom', 'class': GDOptimizer, 'kwargs': kwargs,
                'initial_param': torch.tensor([10.0, 10.0], requires_grad=True)}]
    comparator = OptimizerComparator(configs=configs)
    comparator.compare(is_model=True)
