from torch_optimizer import Lookahead
from optimizer import LookaheadOptimizer, GDOptimizer, SystemOptimizer
import torch
from common import VMOptimizerComparator

if __name__ == '__main__':
    lr = 0.1
    # 内置lookahead.
    initial_param1 = torch.tensor([-10.0, 10.0], requires_grad=True)
    sgd = torch.optim.SGD(params=[initial_param1], lr=lr)  # pytorch内置sgd.
    lookahead = Lookahead(optimizer=sgd, k=5, alpha=0.1)  # 内置lookahead.
    system_optimizer = SystemOptimizer(lookahead)

    # 自定义实现lookahead, version=1.
    initial_param2 = torch.tensor([10.0, 10.0], requires_grad=True)
    lookahead_optimizer_0 = LookaheadOptimizer(optimizer=GDOptimizer(params=initial_param2, lr=lr), p=5,
                                               alpha=0.1, version=0)  # 自定义lookahead.

    # 自定义实现lookahead, version=0.
    initial_param3 = torch.tensor([-10.0, -10.0], requires_grad=True)
    lookahead_optimizer_1 = LookaheadOptimizer(optimizer=GDOptimizer(params=initial_param3, lr=lr), p=5,
                                               alpha=0.1, version=1)  # 自定义lookahead.

    # gd.
    initial_param4 = torch.tensor([10.0, -10.0], requires_grad=True)
    gd = GDOptimizer(params=initial_param4, lr=lr)  # 自定义实现gd.

    optimizers = [system_optimizer, lookahead_optimizer_0, lookahead_optimizer_1, gd]
    comparator = VMOptimizerComparator(optimizers=optimizers)
    comparator.compare()
