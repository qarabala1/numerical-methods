import numpy as np


def get_LU(M: np.ndarray) -> np.ndarray:
    n = M.shape[0]
    L = np.eye(n)
    U = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            U[i, j] = M[i, j] - sum(L[i, k] * U[k, j] for k in range(i))
        for j in range(i, n):
            L[j, i] = (M[j, i] - sum(L[j, k] * U[k, i] for k in range(i))) / U[i, i]

    return L, U


def LU(M: np.ndarray) -> np.ndarray:
    L, U = get_LU(M)
    n = M.shape[0]
    y = np.zeros(n)
    x = np.zeros(n)

    for i in range(n):
        y[i] = M[i, -1] - sum(L[i, j] * y[j] for j in range(i))

    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i, j] * x[j] for j in range(i + 1, n))) / U[i, i]

    return (L, U, x)


def simple_iter(A: np.ndarray, x0: np.ndarray = None, eps: float = 1e-10, max_iter: int = 10**5,):
    n = A.shape[0]
    x = x0.copy() if x0 is not None else np.random.rand(n)
    x = x / np.linalg.norm(x)

    l = 0
    for iter in range(max_iter):
        l_new = 1 / np.max(x)
        x = LU(np.hstack([A, (x * l_new).reshape(-1, 1)]))[2]

        if np.abs(l_new - l) < eps:
            return x, l, iter

        l = l_new

    return x, l, iter


# A = np.array([
#     [2.00, 1.00],
#     [1.00, 2.00]]
# )


A = np.array([[2.2, 1, 0.5, 2],
              [1, 1.3, 2, 1],
              [0.5, 2, 0.5, 1.6],
              [2, 1, 1.6, 2]])

print(A)

eigvec, eigval, iters = simple_iter(A, eps=1e-11, max_iter=10**5)
np_mn_eigval = np.min(np.abs(np.linalg.eigvals(A)))

print(f"\niters: {iters}")
print(f"Собственное значение {eigval}")
print(f"Разница встроенного метода и метода простой итерации: {np.abs(np_mn_eigval - np.abs(eigval))}\n")
print(f"Собственный вектор: {eigvec}")
print((A @ eigvec) / eigval - eigvec)