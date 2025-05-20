from torch_optimizer import Lamb
from common import VMOptimizerComparator
from optimizer import LambOptimizer,SystemOptimizer
import torch

if __name__ == '__main__':
    lr = 0.1
    kwargs1 = {'lr': lr, 'params': [torch.tensor([-10.0, 10.0], requires_grad=True)]}
    kwargs2 = {'lr': lr, 'params': torch.tensor([10.0, 10.0], requires_grad=True)}
    sys_optimizer = SystemOptimizer(optimizer=Lamb(**kwargs1))
    user_optimizer = LambOptimizer(**kwargs2)
    comparator = VMOptimizerComparator(optimizers=[sys_optimizer, user_optimizer])
    comparator.compare()
