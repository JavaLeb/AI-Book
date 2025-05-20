import torch
from common import VMOptimizerComparator, SystemOptimizer
from optimizer import NAGOptimizer, NesterovOptimizer, MomentumOptimizer

if __name__ == '__main__':
    lr = 0.01
    kwargs1 = dict(momentum=0.9, lr=lr, nesterov=True, params=[torch.tensor([-10.0, 10.0], requires_grad=True)])
    kwargs2 = {'momentum': 0.9, 'lr': lr, 'params': torch.tensor([10.0, 10.0], requires_grad=True)}
    kwargs3 = {'momentum': 0.9, 'lr': lr, 'params': torch.tensor([10.0, -10.0], requires_grad=True)}
    kwargs4 = {'momentum': 0.9, 'lr': lr, 'params': torch.tensor([-10.0, -10.0], requires_grad=True)}

    sys_optimizer = SystemOptimizer(optimizer=torch.optim.SGD(**kwargs1))
    user_op2 = NesterovOptimizer(**kwargs2)
    user_op3 = NAGOptimizer(**kwargs3)
    user_op4 = MomentumOptimizer(**kwargs4)

    comparator = VMOptimizerComparator(optimizers=[sys_optimizer, user_op2, user_op3, user_op4])
    comparator.compare()
