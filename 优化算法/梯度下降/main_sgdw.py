from optimizer import SGDOptimizer, SGDWOptimizer, SystemOptimizer
from common import VMOptimizerComparator
import torch

if __name__ == '__main__':
    lr = 0.01
    kwargs1 = {'lr': lr, 'gama': 2.0, 'params': torch.tensor([-10.0, 10.0], requires_grad=True)}
    kwargs2 = {'lr': lr, 'momentum': 0.0, 'gama': 2.0, 'params': torch.tensor([10.0, 10.0], requires_grad=True)}
    user_optimizer1 = optimizer = SGDOptimizer(**kwargs1)
    user_optimizer2 = SGDWOptimizer(**kwargs2)
    comparator = VMOptimizerComparator(optimizers=[user_optimizer1, user_optimizer2])
    comparator.compare()
