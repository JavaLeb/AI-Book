from optimizer import AdaBoundOptimizer, AMSBoundOptimizer, SystemOptimizer
from common import VMOptimizerComparator
from torch_optimizer import AdaBound
import torch

if __name__ == '__main__':
    # AdaBound.
    lr = 5
    kwargs1 = {'lr': lr, 'params': [torch.tensor([-10.0, 10.0], requires_grad=True)]}
    kwargs2 = {'lr': lr, 'params': torch.tensor([10.0, 10.0], requires_grad=True)}
    sys_optimizer = SystemOptimizer(optimizer=AdaBound(**kwargs1))
    user_optimizer = AdaBoundOptimizer(**kwargs2)
    comparator = VMOptimizerComparator(optimizers=[sys_optimizer, user_optimizer])
    comparator.compare()

    # AMSBound.
    lr = 5
    kwargs1 = {'lr': lr, 'amsbound': True, 'params': [torch.tensor([-10.0, 10.0], requires_grad=True)]}
    kwargs2 = {'lr': lr, 'amsbound': True, 'params': torch.tensor([10.0, 10.0], requires_grad=True)}
    sys_optimizer = SystemOptimizer(optimizer=AdaBound(**kwargs1))
    user_optimizer = AdaBoundOptimizer(**kwargs2)
    comparator = VMOptimizerComparator(optimizers=[sys_optimizer, user_optimizer])
    comparator.compare()
