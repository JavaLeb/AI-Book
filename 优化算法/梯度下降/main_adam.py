import torch
from optimizer import AdamOptimizer, SystemOptimizer
from common import VMOptimizerComparator

if __name__ == '__main__':
    lr = 5
    kwargs1 = {'lr': lr, 'params': [torch.tensor([-10.0, 10.0], requires_grad=True)]}
    kwargs2 = {'lr': lr, 'params': torch.tensor([10.0, 10.0], requires_grad=True)}
    sys_optimizer = SystemOptimizer(optimizer=torch.optim.Adam(**kwargs1))
    user_op = AdamOptimizer(**kwargs2)

    comparator = VMOptimizerComparator(optimizers=[sys_optimizer, user_op])
    comparator.compare()
