import torch
from optimizer import AdaBeliefOptimizer, SystemOptimizer
from common import VMOptimizerComparator
from pytorch_optimizer import AdaBelief

if __name__ == '__main__':
    lr = 1
    kwargs1 = {'lr': lr, 'params': [torch.tensor([-10.0, 10.0], requires_grad=True)]}
    kwargs2 = {'lr': lr, 'params': torch.tensor([10.0, 10.0], requires_grad=True)}
    sys_optimizer = SystemOptimizer(optimizer=AdaBelief(**kwargs1))
    user_optimizer = AdaBeliefOptimizer(**kwargs2)
    comparator = VMOptimizerComparator(optimizers=[sys_optimizer, user_optimizer])
    comparator.compare()
