import torch
from common import LinearModel
from common import generate_data
from optimizer import StochasticGradientDescent
from common import MSELoss

if __name__ == '__main__':
    X, y = generate_data()
    init_params = torch.tensor([10.0, 10.0], requires_grad=True)
    for batch_size in range(10, 0, -1):  # batch_size的顺序不同，结果也不同.
        model = LinearModel(
            in_features=1,
            out_features=1,
            optimizer=StochasticGradientDescent(loss=MSELoss(), params=init_params, lr=0.01)
        )
        model.fit(X, y, epochs=2000, batch_size=batch_size, shuffled=False)
