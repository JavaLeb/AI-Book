import torch
from optimizer import SophiaHOptimizer, SystemOptimizer
from common import VMOptimizerComparator
from pytorch_optimizer import SophiaH

if __name__ == '__main__':
    lr = 0.1
    kwargs1 = {'lr': lr, 'num_samples': 10, 'params': [torch.tensor([-10.0, 10.0], requires_grad=True)]}
    kwargs2 = {'lr': lr, 'num_samples': 10, 'params': torch.tensor([10.0, 10.0], requires_grad=True)}
    sys_optimizer = SystemOptimizer(optimizer=SophiaH(**kwargs1))
    user_optimizer = SophiaHOptimizer(**kwargs2)
    comparator = VMOptimizerComparator(optimizers=[sys_optimizer, user_optimizer])
    comparator.compare()
