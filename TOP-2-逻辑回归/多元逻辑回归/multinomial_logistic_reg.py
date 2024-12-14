from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
from sklearn.linear_model import LogisticRegression

X = iris.data.values
y = iris['target']

model = LogisticRegression()
model.fit(X, y)
test = X[0,:]
prob =model.predict_proba(test.reshape(1, -1))
print(prob)
print()
