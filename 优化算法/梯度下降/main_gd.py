import torch
from common import VMOptimizerComparator, LROptimizerComparator
from optimizer import GDOptimizer, SystemOptimizer

if __name__ == '__main__':
    kwargs1 = {'lr': 0.01, 'params': [torch.tensor([-10.0, 10.0], requires_grad=True)]}
    kwargs2 = {'lr': 0.01, 'params': torch.tensor([10.0, 10.0], requires_grad=True)}
    sys_optimizer = SystemOptimizer(optimizer=torch.optim.SGD(**kwargs1))
    user_optimizer = GDOptimizer(**kwargs2)
    comparator = VMOptimizerComparator(optimizers=[sys_optimizer, user_optimizer])
    comparator.compare()

    # 动态学习率.
    kwargs1 = {'lr': 0.1, 'dynamic': False, 'params': torch.tensor([-10.0, 10.0], requires_grad=True)}
    kwargs2 = {'lr': 0.1, 'dynamic': True, 'params': torch.tensor([10.0, 10.0], requires_grad=True)}
    user_optimizer1 = GDOptimizer(**kwargs1)
    user_optimizer2 = GDOptimizer(**kwargs2)
    comparator = VMOptimizerComparator(optimizers=[user_optimizer1, user_optimizer2])
    comparator.compare()

    # 不同学习率.
    kwargs1 = {'lr': 0.5, 'params': torch.tensor([-10.0, 10.0], requires_grad=True)}
    kwargs2 = {'lr': 0.45, 'params': torch.tensor([10.0, 10.0], requires_grad=True)}
    kwargs3 = {'lr': 0.1, 'params': torch.tensor([-10.0, -10.0], requires_grad=True)}
    user_op1 = GDOptimizer(**kwargs1)
    user_op2 = GDOptimizer(**kwargs2)
    user_op3 = GDOptimizer(**kwargs3)
    comparator = VMOptimizerComparator(optimizers=[user_op1, user_op2, user_op3])
    comparator.compare()

    # 线性回归.
    kwargs1 = {'lr': 0.05, 'params': [torch.tensor([10.0, 10.0], requires_grad=True)]}
    kwargs2 = {'lr': 0.05, 'params': torch.tensor([10.0, 10.0], requires_grad=True)}
    sys_optimizer = SystemOptimizer(optimizer=torch.optim.SGD(**kwargs1))
    user_optimizer = GDOptimizer(**kwargs2)
    comparator = LROptimizerComparator(optimizers=[sys_optimizer, user_optimizer])
    comparator.compare()
