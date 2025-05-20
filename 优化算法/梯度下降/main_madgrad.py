import torch_optimizer
from optimizer import MADGRADOptimizer,SystemOptimizer
from common import VMOptimizerComparator
import torch


if __name__ == '__main__':
    lr = 10
    kwargs1 = {'lr': lr, 'params': [torch.tensor([-10.0, 10.0], requires_grad=True)]}
    kwargs2 = {'lr': lr, 'params': torch.tensor([10.0, 10.0], requires_grad=True)}
    sys_optimizer = SystemOptimizer(optimizer=torch_optimizer.MADGRAD(**kwargs1))
    user_optimizer = MADGRADOptimizer(**kwargs2)
    comparator = VMOptimizerComparator(optimizers=[sys_optimizer, user_optimizer])
    comparator.compare()