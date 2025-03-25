from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

X = [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]]
y = [0, 1, 1, 0]

reg = MLPRegressor(hidden_layer_sizes=[2, 1],
                   batch_size=4,
                   activation='logistic',
                   learning_rate_init=0.1,
                   random_state=42)

reg.fit(X, y)
y_predict = reg.predict(X)
print('y_predict=', y_predict)
mse = mean_squared_error(y, y_predict)
print('mes=', mse)

cls = MLPClassifier(hidden_layer_sizes=[2, 1],
                    batch_size=4,
                    activation='logistic',
                    learning_rate_init=0.0001,
                    max_iter=20000,
                    random_state=42)
cls.fit(X, y)
y_pred = cls.predict(X)
print('y_pred=', y_pred)
accuracy = accuracy_score(y, y_pred)
print('accuracy=', accuracy)
