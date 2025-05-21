import torch
from optimizer import AdaGradOptimizer, SystemOptimizer
from common import VMOptimizerComparator

if __name__ == '__main__':
    lr = 3.0
    kwargs1 = dict(lr=lr, params=[torch.tensor([-10.0, 10.0], requires_grad=True)])
    kwargs2 = dict(lr=lr, version='vector', params=torch.tensor([10.0, 10.0], requires_grad=True))
    kwargs3 = dict(lr=lr, version='diag', params=torch.tensor([-10.0, -10.0], requires_grad=True))
    kwargs4 = dict(lr=lr, version='full', params=torch.tensor([10.0, -10.0], requires_grad=True))
    sys_optimizer = SystemOptimizer(optimizer=torch.optim.Adagrad(**kwargs1))
    user_op2 = AdaGradOptimizer(**kwargs2)
    user_op3 = AdaGradOptimizer(**kwargs3)
    user_op4 = AdaGradOptimizer(**kwargs4)
    comparator = VMOptimizerComparator(optimizers=[sys_optimizer, user_op2, user_op3, user_op4])
    comparator.compare()
