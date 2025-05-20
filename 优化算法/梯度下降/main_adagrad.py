import torch
from optimizer import AdaGradOptimizer, SystemOptimizer
from common import VMOptimizerComparator

if __name__ == '__main__':
    lr = 1
    kwargs1 = dict(lr=lr, params=[torch.tensor([-10.0, 10.0], requires_grad=True)])
    kwargs2 = dict(lr=lr, params=torch.tensor([10.0, 10.0], requires_grad=True))
    sys_optimizer = SystemOptimizer(optimizer=torch.optim.Adagrad(**kwargs1))
    user_op = AdaGradOptimizer(**kwargs2)

    comparator = VMOptimizerComparator(optimizers=[sys_optimizer, user_op])
    comparator.compare()
