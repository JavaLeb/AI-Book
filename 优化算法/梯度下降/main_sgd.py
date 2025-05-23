import torch
from common import LROptimizerComparator
from optimizer import SGDOptimizer, SystemOptimizer

if __name__ == '__main__':
    lr = 0.01
    kwargs1 = {'lr': lr, 'params': [torch.tensor([10.0, 10.0], requires_grad=True)]}
    kwargs2 = {'lr': lr, 'params': torch.tensor([10.0, 10.0], requires_grad=True)}
    sys_optimizer = SystemOptimizer(optimizer=torch.optim.SGD(**kwargs1))
    user_optimizer = SGDOptimizer(**kwargs2)
    comparator = LROptimizerComparator(optimizers=[sys_optimizer, user_optimizer], batch_size=1)
    comparator.compare()
