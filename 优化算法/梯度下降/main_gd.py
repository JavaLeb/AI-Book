import torch
from common import OptimizerComparator
from optimizer import GDOptimizer

if __name__ == '__main__':
    kwargs = {'lr': 0.01}
    configs = [{'type': 'system', 'class': torch.optim.SGD, 'kwargs': kwargs},
               {'type': 'custom', 'class': GDOptimizer, 'kwargs': kwargs}]
    comparator = OptimizerComparator(configs=configs)
    comparator.compare()

    # 动态学习率.
    kwargs1 = {'lr': 0.1, 'dynamic': False, 'params': torch.tensor([10.0, 10.0], requires_grad=True)}
    kwargs2 = {'lr': 0.1, 'dynamic': True, 'params': torch.tensor([-10.0, 10.0], requires_grad=True)}
    configs = [{'type': 'custom', 'class': GDOptimizer, 'kwargs': kwargs1},
               {'type': 'custom', 'class': GDOptimizer, 'kwargs': kwargs2}]
    comparator = OptimizerComparator(configs=configs)
    comparator.compare()

    # 线性回归.
    lr = 0.01
    kwargs1 = {'lr': lr, 'params': [torch.tensor([10.0, 10.0], requires_grad=True)]}
    kwargs2 = {'lr': lr, 'params': torch.tensor([10.0, 10.0], requires_grad=True)}
    configs = [{'type': 'system', 'class': torch.optim.SGD, 'kwargs': kwargs1},
               {'type': 'custom', 'class': GDOptimizer, 'kwargs': kwargs2}]
    comparator = OptimizerComparator(configs=configs)
    comparator.compare(is_model=True)
