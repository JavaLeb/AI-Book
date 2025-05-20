import torch
from optimizer import AMSGradOptimizer, SystemOptimizer
from common import VMOptimizerComparator

if __name__ == '__main__':
    lr = 5
    kwargs1 = {'lr': lr, 'amsgrad': True, 'params': [torch.tensor([-10.0, 10.0], requires_grad=True)]}
    kwargs2 = {'lr': lr, 'momentum': 0.9, 'decay_factor': 0.999,
               'params': torch.tensor([10.0, 10.0], requires_grad=True)}
    sys_optimizer = SystemOptimizer(optimizer=torch.optim.Adam(**kwargs1))
    user_op = AMSGradOptimizer(**kwargs2)

    comparator = VMOptimizerComparator(optimizers=[sys_optimizer, user_op])
    comparator.compare()
