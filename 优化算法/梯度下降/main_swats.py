from optimizer import SWATSOptimizer, SystemOptimizer
from torch_optimizer import SWATS
from common import VMOptimizerComparator
import torch

if __name__ == '__main__':
    lr = 5
    kwargs1 = {'lr': lr, 'params': [torch.tensor([-10.0, 10.0], requires_grad=True)]}
    kwargs2 = {'lr': lr, 'params': torch.tensor([10.0, 10.0], requires_grad=True)}

    sys_optimizer = SystemOptimizer(optimizer=SWATS(**kwargs1))
    user_optimizer = SWATSOptimizer(**kwargs2)

    comparator = VMOptimizerComparator(optimizers=[sys_optimizer, user_optimizer])
    comparator.compare()
