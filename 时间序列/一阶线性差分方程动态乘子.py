import numpy as np
from matplotlib import pyplot as plt

x = np.arange(0, 21)
alphas = [0.9, -0.9, 1, -1, 1.1, -1.1]

for i in range(0, len(alphas)):
    if i % 4 == 0:
        plt.figure()
    y = np.power(alphas[i], x)
    plt.subplot(2, 2, i % 4 + 1)
    plt.bar(x, y, width=0.5)
    plt.title(f'$\\alpha_1={alphas[i]}$')
    plt.xlabel(f'$h$')
    plt.ylabel('${\\alpha_1}^h$')
plt.show()
