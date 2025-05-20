import torch
from common import VMOptimizerComparator, SystemOptimizer
from optimizer import MomentumOptimizer

if __name__ == '__main__':
    kwargs1 = {'momentum': 0.9, 'lr': 0.01, 'params': [torch.tensor([-10.0, 10.0], requires_grad=True)]}
    kwargs2 = {'momentum': 0.9, 'lr': 0.01, 'params': torch.tensor([10.0, 10.0], requires_grad=True)}
    sys_optimizer = SystemOptimizer(optimizer=torch.optim.SGD(**kwargs1))
    user_optimizer = MomentumOptimizer(**kwargs2)
    comparator = VMOptimizerComparator(optimizers=[sys_optimizer, user_optimizer])
    comparator.compare()

    kwargs1 = {'momentum': 0.9, 'version': 0, 'lr': 0.01, 'params': torch.tensor([-10.0, 10.0], requires_grad=True)}
    kwargs2 = {'momentum': 0.9, 'version': 1, 'lr': 0.01, 'params': torch.tensor([10.0, -10.0], requires_grad=True)}
    kwargs3 = {'momentum': 0.9, 'version': 2, 'lr': 0.01, 'params': torch.tensor([10.0, 10.0], requires_grad=True)}
    kwargs4 = {'momentum': 0.9, 'version': 3, 'lr': 0.01, 'params': torch.tensor([-10.0, -10.0], requires_grad=True)}
    user_op1 = MomentumOptimizer(**kwargs1)
    user_op2 = MomentumOptimizer(**kwargs2)
    user_op3 = MomentumOptimizer(**kwargs3)
    user_op4 = MomentumOptimizer(**kwargs4)
    comparator = VMOptimizerComparator(optimizers=[user_op1, user_op2, user_op3, user_op4])
    comparator.compare()
