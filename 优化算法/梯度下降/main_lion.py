import torch
from optimizer import LionOptimizer, SystemOptimizer
from common import VMOptimizerComparator
from pytorch_optimizer import Lion

if __name__ == '__main__':
    lr = 0.03
    kwargs1 = {'lr': lr, 'r': 0.0, 'betas': (0.9, 0.99), 'weight_decay': 0.1,
               'params': [torch.tensor([-10.0, 10.0], requires_grad=True)]}
    kwargs2 = {'lr': lr, 'weight_decay': 0.1, 'params': torch.tensor([10.0, 10.0], requires_grad=True)}
    sys_optimizer = SystemOptimizer(optimizer=Lion(**kwargs1))
    user_optimizer = LionOptimizer(**kwargs2)
    comparator = VMOptimizerComparator(optimizers=[sys_optimizer, user_optimizer])
    comparator.compare()
