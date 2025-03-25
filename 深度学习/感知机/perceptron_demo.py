from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

X = [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]]
y = [0, 0, 0, 1]
clf = Perceptron(random_state=42, max_iter=1000, eta0=1.0, tol=1e-3)
clf.fit(X, y)
y_hat = clf.predict(X)
print('y_hat=', y_hat)
accuracy = accuracy_score(y, y_hat)
print('accuracy=', accuracy)
print(f'y={clf.coef_}x+{clf.intercept_}')
print('n_iter_=', clf.n_iter_)
