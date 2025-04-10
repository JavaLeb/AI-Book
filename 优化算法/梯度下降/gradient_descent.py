class GradientDescent:
    """
    梯度下降法.
    """

    def __init__(self, params, lr=0.01, tol=1e-6):
        """
        初始化方法.
        :param params: 参数列表（可以多组参数）.
        :param lr: 学习率.
        :param tol: 误差阈值.
        """
        self.lr = lr
        self.tol = tol
        self.params = params

    def step(self):
        """
        参数更新步骤.
        :return:
        """
        if self.params.grad is not None:
            self.params.data -= self.lr * self.params.grad.data  # 参数更新.
            self.params.grad.zero_()  # 清空梯度.


class StochasticGradientDescent:
    """
    随机梯度下降法：实现与梯度下降相同，不同在于训练时，样本损失不同.
    """

    def __init__(self, params, lr=0.01, tol=1e-6):
        self.lr = lr
        self.tol = tol
        self.params = params

    def step(self):
        if self.params.grad is not None:
            self.params.data -= self.lr * self.params.grad.data  # 自定义更新规则
            self.params.grad.zero_()
