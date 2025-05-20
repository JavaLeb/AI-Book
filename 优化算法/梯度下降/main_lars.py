from torch_optimizer import LARS
from common import VMOptimizerComparator
from optimizer import LARSOptimizer, SystemOptimizer
import torch

if __name__ == '__main__':
    lr = 0.1
    kwargs1 = {'lr': lr, 'momentum': 0.9, 'weight_decay': 0.01, 'trust_coefficient': 0.01,
               'params': [torch.tensor([-10.0, 10.0], requires_grad=True)]}
    kwargs2 = {'lr': lr, 'momentum': 0.9, 'weight_decay': 0.01, 'lars_c': 0.01,
               'params': torch.tensor([10.0, 10.0], requires_grad=True)}
    sys_optimizer = SystemOptimizer(optimizer=LARS(**kwargs1))
    user_optimizer = LARSOptimizer(**kwargs2)
    comparator = VMOptimizerComparator(optimizers=[sys_optimizer, user_optimizer])
    comparator.compare()


