import torch
from common import VMOptimizerComparator, SystemOptimizer
from optimizer import RMSPropOptimizer

if __name__ == '__main__':
    lr = 0.5
    kwargs1 = {'lr': lr, 'params': [torch.tensor([-10.0, 10.0], requires_grad=True)]}
    kwargs2 = {'lr': lr, 'params': torch.tensor([10.0, 10.0], requires_grad=True)}
    kwargs3 = {'lr': lr, 'decay_factor': 0.3, 'params': torch.tensor([-10.0, -10.0], requires_grad=True)}
    sys_optimizer = SystemOptimizer(optimizer=torch.optim.RMSprop(**kwargs1))
    user_op2 = RMSPropOptimizer(**kwargs2)
    user_op3 = RMSPropOptimizer(**kwargs3)
    comparator = VMOptimizerComparator(optimizers=[sys_optimizer, user_op2, user_op3])
    comparator.compare()
