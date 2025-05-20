from optimizer import QHAdamOptimizer, SystemOptimizer
from torch_optimizer import QHAdam
from common import VMOptimizerComparator
import torch

if __name__ == '__main__':
    lr = 5
    kwargs1 = {'lr': lr, 'params': [torch.tensor([-10.0, 10.0], requires_grad=True)]}
    kwargs2 = {'lr': lr, 'params': torch.tensor([10.0, 10.0], requires_grad=True)}
    sys_optimizer = SystemOptimizer(optimizer=QHAdam(**kwargs1))
    user_optimizer = QHAdamOptimizer(**kwargs2)
    comparator = VMOptimizerComparator(optimizers=[sys_optimizer, user_optimizer])
    comparator.compare()
