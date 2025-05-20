import torch
from common import OptimizerComparator
from optimizer import GDOptimizer, SystemOptimizer

if __name__ == '__main__':
    kwargs1 = {'lr': 0.01, 'params': [torch.tensor([-10.0, 10.0], requires_grad=True)]}
    kwargs2 = {'lr': 0.01, 'params': torch.tensor([10.0, 10.0], requires_grad=True)}
    sys_optimizer = SystemOptimizer(optimizer=torch.optim.SGD(**kwargs1))
    user_optimizer = GDOptimizer(**kwargs2)
    comparator = OptimizerComparator()
    comparator.compare(optimizers=[sys_optimizer, user_optimizer])

    # 动态学习率.
    kwargs1 = {'lr': 0.1, 'dynamic': False, 'params': torch.tensor([10.0, 10.0], requires_grad=True)}
    kwargs2 = {'lr': 0.1, 'dynamic': True, 'params': torch.tensor([-10.0, 10.0], requires_grad=True)}
    user_optimizer1 = GDOptimizer(**kwargs1)
    user_optimizer2 = GDOptimizer(**kwargs2)
    comparator = OptimizerComparator()
    comparator.compare(optimizers=[user_optimizer1, user_optimizer2])

    # 线性回归.
    kwargs1 = {'lr': 0.01, 'params': [torch.tensor([10.0, 10.0], requires_grad=True)]}
    kwargs2 = {'lr': 0.01, 'params': torch.tensor([10.0, 10.0], requires_grad=True)}
    sys_optimizer = SystemOptimizer(optimizer=torch.optim.SGD(**kwargs1))
    user_optimizer = GDOptimizer(**kwargs2)
    comparator = OptimizerComparator()
    comparator.compare(optimizers=[sys_optimizer, user_optimizer], is_model=True)
