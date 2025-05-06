import torch
from optimizer import AdamWOptimizer
from common import OptimizerComparator

if __name__ == '__main__':
    kwargs = {'lr': 0.3}
    configs = [{'type': 'system', 'class': torch.optim.AdamW, 'kwargs': kwargs},
               {'type': 'custom', 'class': AdamWOptimizer, 'kwargs': kwargs}]
    comparator = OptimizerComparator(configs=configs)
    comparator.compare()
