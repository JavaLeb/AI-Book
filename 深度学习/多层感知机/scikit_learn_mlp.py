from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np

# 异或问题的输入样本
X = [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]]
y = [0, 1, 1, 0]

# 初始化MLP回归器.
reg = MLPRegressor(hidden_layer_sizes=(2,),
                   batch_size=2,
                   solver="lbfgs",
                   activation='logistic',
                   learning_rate_init=0.1,
                   random_state=20)

reg.fit(X, y)  # 拟合数据
y_predict = reg.predict(X)  # 预测.
mse = mean_squared_error(y, y_predict)  # 计算均方误差.
print('y_predict=', np.round(y_predict, 2), ', mes=', np.round(mse, 2))

# 初始化MLP分类器.
cls = MLPClassifier(hidden_layer_sizes=(2,),
                    batch_size=2,
                    solver="lbfgs",
                    activation='logistic',
                    learning_rate_init=0.1,
                    random_state=20)
cls.fit(X, y)  # 拟合数据.
y_pred = cls.predict(X)  # 预测.
accuracy = accuracy_score(y, y_pred)
print('y_pred=', y_pred, ', accuracy=', accuracy)
