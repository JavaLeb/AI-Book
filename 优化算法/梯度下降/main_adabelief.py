import torch
from optimizer import AdaBeliefOptimizer
from common import OptimizerComparator
from pytorch_optimizer import AdaBelief

if __name__ == '__main__':
    lr = 1
    kwargs1 = {'lr': lr, 'params': [torch.tensor([-10.0, 10.0], requires_grad=True)]}
    kwargs2 = {'lr': lr, 'params': torch.tensor([10.0, 10.0], requires_grad=True)}
    configs = [
        {'type': 'system', 'class': AdaBelief, 'kwargs': kwargs1},
        {'type': 'custom', 'class': AdaBeliefOptimizer, 'kwargs': kwargs2}
    ]
    comparator = OptimizerComparator(configs=configs)
    comparator.compare()
