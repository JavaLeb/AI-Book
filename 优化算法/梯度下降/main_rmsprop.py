import torch
from common import VMOptimizerComparator,SystemOptimizer
from optimizer import RMSPropOptimizer

if __name__ == '__main__':
    lr = 1.0
    kwargs1 = {'lr': lr, 'params': [torch.tensor([-10.0, 10.0], requires_grad=True)]}
    kwargs2 = {'lr': lr, 'params': torch.tensor([10.0, 10.0], requires_grad=True)}
    sys_optimizer = SystemOptimizer(optimizer=torch.optim.RMSprop(**kwargs1))
    user_optimizer = RMSPropOptimizer(**kwargs2)
    comparator = VMOptimizerComparator(optimizers=[sys_optimizer, user_optimizer])
    comparator.compare()
