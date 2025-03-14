import numpy as np


def mu(r, w):
    return np.dot(r, w @ w.T @ r) / np.dot(w @ w.T @ r, w @ w.T @ r)


def r(A, b, x):
    return np.dot(A, x) - b


def gradient_descent(A, b, tol=1e-1, max_iterations=1000000):
    x = np.zeros_like(b)
    for i in range(max_iterations):
        x_old = x.copy()
        r_k = r(A, b, x)
        mu_k = mu(r_k, A)
        x = x_old - mu_k * A.T @ r_k

        if np.linalg.norm(x - x_old) < tol:
            return x, i

    raise ValueError(
        f"Convergence not achieved after {max_iterations} iterations")



A = np.array([
    [11, 1.2, 2.1, 0.9],
    [1.2, 12, 1.5, 2.5],
    [2.1, 1.5, 9.8, 1.3],
    [0.9, 2.5, 1.3, 12.1]
])

b = np.array([1, 2, 3, 4])

x_star, i = gradient_descent(A, b)
print("Решение системы:", x_star)
print("За ", i, " итераций")
print("Погрешность", (A @ x_star) - b)
print("Встроенный метод: ", np.linalg.solve(A, b))


