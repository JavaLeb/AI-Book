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
        for param in self.params:
            if param.grad is not None:
                param.data -= self.lr * param.grad.data  # 参数更新.
                param.grad.zero_()  # 清空梯度.


