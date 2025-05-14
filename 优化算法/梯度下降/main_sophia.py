import torch
from optimizer import SophiaHOptimizer
from common import OptimizerComparator
from pytorch_optimizer import SophiaH

if __name__ == '__main__':
    lr = 1
    kwargs1 = {'lr': lr, 'num_samples': 10, 'params': [torch.tensor([-10.0, 10.0], requires_grad=True)]}
    kwargs2 = {'lr': lr, 'b': 10, 'params': torch.tensor([10.0, 10.0], requires_grad=True)}
    configs = [
        {'type': 'system', 'class': SophiaH, 'kwargs': kwargs1},
        {'type': 'custom', 'class': SophiaHOptimizer, 'kwargs': kwargs2}
    ]
    comparator = OptimizerComparator(configs=configs)
    comparator.compare()
