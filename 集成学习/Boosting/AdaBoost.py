import numpy as np

# [0 1 2 3 4 5 6 7 8 9]
x = np.arange(0, 10)
y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])

# [0 1 2 3 4 5 6 7 8 9]
thetas = np.arange(-1, 10)


def h(w, H_x, z):
    min_e = 1
    min_theta = 0
    min_hx = 0
    e = []
    for theta in thetas:
        for v in [1]:
            hx = np.where(x <= theta, v, -v)
            L = np.sum(w * np.where(hx == y, 0, 1))
            e.append(L)
            if L < min_e:
                min_theta = theta
                min_e = L
                min_hx = hx
    print('min_theta=', min_theta, ',min_hx=', min_hx, ',min_e=', min_e, 'e=', np.round(e, 4))
    alpha = 1 / 2 * np.log((1 - min_e) / min_e)
    print('alpha=', alpha)
    z_t = 2 * np.sqrt(min_e * (1 - min_e))
    print('z_t=', z_t)
    z *= z_t
    print('z=', z)
    equal_mask = min_hx == y  # 找到相等的元素的位置
    not_equal_mask = min_hx != y  # 找到不相等的元素的位置
    increase_ratio = 1 / (2 * (1 - min_e))
    decrease_ratio = 1 / (2 * min_e)
    w[equal_mask] = w[equal_mask] * increase_ratio
    w[not_equal_mask] = w[not_equal_mask] * decrease_ratio
    print('w=', w)
    H_x = H_x + alpha * min_hx
    print('H_x=', H_x)
    F_x = np.where(H_x >= 0, 1, -1)
    print('F_x=', F_x)
    error = np.sum(np.where(F_x == y, 0, 1)) / len(y)
    print('error=', error)
    return (w, H_x, z)


# [0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1]
w = np.full(10, 0.1)
T = 20000
H_x = np.full(10, 0)
z = 1
for t in np.arange(1, T):
    print(f'------------------t={t}-------------------------------')
    (w, H_x, z) = h(w, H_x, z)
    print(f'------------------t={t}-------------------------------')
