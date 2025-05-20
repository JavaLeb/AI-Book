import torch
from optimizer import AdamWOptimizer, SystemOptimizer
from common import VMOptimizerComparator

if __name__ == '__main__':
    lr = 0.3
    kwargs1 = {'lr': lr, 'params': [torch.tensor([-10.0, 10.0], requires_grad=True)]}
    kwargs2 = {'lr': lr, 'params': torch.tensor([10.0, 10.0], requires_grad=True)}
    sys_optimizer = SystemOptimizer(optimizer=torch.optim.AdamW(**kwargs1))
    user_optimizer = AdamWOptimizer(**kwargs2)
    comparator = VMOptimizerComparator(optimizers=[sys_optimizer, user_optimizer])
    comparator.compare()
