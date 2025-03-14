import numpy as np

def get_LU(A):
    n = A.shape[0]
    LU = np.zeros((n, n))

    for k in range(n):
        for j in range(k, n):  # Вычисляем элементы U
            LU[k, j] = A[k, j] - LU[k, :k] @ LU[:k, j]
        for i in range(k + 1, n):  # Вычисляем элементы L
            LU[i, k] = (A[i, k] - LU[i, :k] @ LU[:k, k]) / LU[k, k]
    
    # Получение L и U из LU
    L = np.tril(LU, -1) + np.eye(n)  # Нижняя треугольная + единицы на диагонали
    U = np.triu(LU)  # Верхняя треугольная
    print("L матрица:\n", L)
    print("U матрица:\n", U)
    return L, U

def solve_LU(L, U, b):
    n = L.shape[0]
    y = np.zeros(n)

    # Решение Ly = b
    for i in range(n):
        y[i] = b[i] - L[i, :i] @ y[:i]

    x = np.zeros(n)

    # Решение Ux = y
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]

    return x

if __name__ == '__main__':
    A = np.array([
        [0.411, 0.421, -0.333, 0.313, -0.141, -0.381, 0.245],
        [0.241, 0.705, 0.139, -0.409, 0.321, 0.0625, 0.101],
        [0.123, -0.239, 0.502, 0.901, 0.243, 0.819, 0.321],
        [0.413, 0.309, 0.801, 0.865, 0.423, 0.118, 0.183],
        [0.241, -0.221, -0.243, 0.134, 1.274, 0.712, 0.423],
        [0.281, 0.525, 0.719, 0.118, -0.974, 0.808, 0.923],
        [0.246, -0.301, 0.231, 0.813, -0.702, 1.223, 1.105]
    ])
    b = np.array([0.096, 1.252, 1.024, 1.023, 1.155, 1.937, 1.673])

    # Разложение на L и U
    L, U = get_LU(A)

    # Решение системы
    solution = solve_LU(L, U, b)
    print("Решение системы уравнений:", solution)
    print(A @ solution - b)
    print("Погрешность: ", L @ U - A)
    