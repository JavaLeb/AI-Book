import torch
from optimizer import NewtonOptimizer, SystemOptimizer
from common import VMOptimizerComparator

if __name__ == "__main__":
    lr = 0.1
    kwargs1 = {'lr': lr, 'max_iter': 2000, 'params': [torch.tensor([-10.0, 10.0], requires_grad=True)]}
    kwargs2 = {'lr': lr, 'params': torch.tensor([10.0, 10.0], requires_grad=True)}
    sys_optimizer = SystemOptimizer(optimizer=torch.optim.LBFGS(**kwargs1))
    user_optimizer = NewtonOptimizer(**kwargs2)
    comparator = VMOptimizerComparator(optimizers=[sys_optimizer, user_optimizer])
    comparator.compare()
