from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.utils.validation import check_X_y


class RosenblattPerceptron:
    def __init__(self, learning_rate=1.0, max_iter=10000):
        self.eta = learning_rate  # 学习率 η
        self.max_iter = max_iter  # 最大迭代次数
        self.weights = None  # 权重向量
        self.bias = 0  # 偏置

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        n_samples, n_features = X.shape  # 获取样本数，特征数.
        y = np.where(y == 0, -1, 1)  # 标签转换为{-1,1}.
        self.weights = np.zeros(n_features)  # 初始化权重.
        self.iter_n = 0
        for _ in range(self.max_iter):
            self.iter_n += 1
            misclassified = False
            for idx, xi in enumerate(X):
                z = np.dot(xi, self.weights) + self.bias
                yi_hat = 1 if z >= 0 else -1

                if y[idx] * yi_hat <= 0:  # 误分类
                    self.weights += self.eta * (y[idx] - yi_hat) * xi
                    self.bias += self.eta * (y[idx] - yi_hat)
                    misclassified = True

            if not misclassified:  # 全部分类正确
                break

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias

        return np.where(z >= 0, 1, 0)


X = [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]]
y = [0, 0, 0, 1]
clf = RosenblattPerceptron(max_iter=1000, learning_rate=1.0)
clf.fit(X, y)
y_hat = clf.predict(X)
print('y_hat=', y_hat)
accuracy = accuracy_score(y, y_hat)
print('accuracy=', accuracy)
print(f'y={clf.weights}x+{clf.bias}')
print('iter_n=', clf.iter_n)
