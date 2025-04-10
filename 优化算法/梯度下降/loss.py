import torch
from torch import Tensor


class SquareFunction:
    """
    函数f(x,y) = c1 * x1^2 + c2 * x2^2.
    """

    def __init__(self, c: Tensor) -> None:
        """

        :param c:   shape=Tensor(n,)
        """
        self.c = c

    def forward(self, params: Tensor) -> Tensor:
        """

        :param params: shape=Tensor(n,)
        :return: 沿最后一个维度求和. 例如shape(n,)只有一个维度，求和后维度为shape=Tensor(),零维（标量）.
        """
        return torch.sum(self.c * params ** 2, dim=-1)


class MSELoss:
    def __init__(self) -> None:
        pass

    def forward(self, y_predict: Tensor, y_true: Tensor) -> Tensor:
        return torch.mean((y_predict - y_true) ** 2)
