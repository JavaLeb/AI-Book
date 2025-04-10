import torch
from common import SGDLinearModel
from common import generate_data
from gradient_descent import StochasticGradientDescent
from common import MSELoss

if __name__ == '__main__':
    X, y = generate_data()
    init_params = torch.tensor([10.0, 10.0], requires_grad=True)
    for batch_size in range(10, 0, -2):  # batch_size的顺序不同，结果也不同.
        model = SGDLinearModel(
            in_features=1,
            out_features=1,
            loss=MSELoss(),
            optimizer=StochasticGradientDescent(params=init_params, lr=0.01)
        )
        model.fit(X, y, epochs=20000, batch_size=batch_size, shuffled=False)
