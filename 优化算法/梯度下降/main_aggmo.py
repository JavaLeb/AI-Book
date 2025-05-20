from optimizer import AggMoOptimizer
from torch_optimizer import AggMo
from common import VMOptimizerComparator, SystemOptimizer
import torch

if __name__ == '__main__':
    kwargs1 = dict(lr=0.5, params=[torch.tensor([-10.0, 10.0], requires_grad=True)])
    kwargs2 = dict(lr=0.5, params=torch.tensor([10.0, 10.0], requires_grad=True))
    sys_optimizer = SystemOptimizer(optimizer=AggMo(**kwargs1))
    user_op = AggMoOptimizer(**kwargs2)

    comparator = VMOptimizerComparator(optimizers=[sys_optimizer, user_op])
    comparator.compare()
