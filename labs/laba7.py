import numpy as np


def sigma(a):
    return -1 if a < 0 else 1


def kd(i, j):  
    return 1 if i == j else 0


def qr(A):
    n = A.shape[0]
    R = np.copy(A)
    Q = np.eye(n)

    for k in range(n - 1):
        # Шаг 1: Формируем вектор отражения ps
        ps = np.zeros(n)
        norm_x = np.sqrt(np.sum(R[k:, k] ** 2))
        ps[k] = R[k, k] + sigma(R[k, k]) * norm_x
        ps[k + 1:] = R[k + 1:, k]

        # Формируем матрицу P_k 
        P_k = np.eye(n) - 2 * np.outer(ps, ps) / np.sum(ps[k:] ** 2)

        # Шаг 3: Применяем P_k к R и Q
        R = P_k @ R
        print(R)
        Q = P_k @ Q

    return Q.T, R


def get_answer(A, b):
    Q, R = qr(A)

    # Вычисляем Q^T * b
    b_tilde = np.dot(Q.T, b)

    n = R.shape[1]
    x = np.zeros(n)

    # Метод обратной подстановки для решения Rx = Q^T b
    for i in range(n - 1, -1, -1):
        x[i] = (b_tilde[i] - np.dot(R[i, i + 1:], x[i + 1:])) / R[i, i]

    return x


def main():
    # A = np.array([
    #     [1.2, 4.1, -6.2, 0.1], 
    #     [4.5, -6.1, 3, -2], 
    #     [0.1, 5, 2.1, -3],
    #     [3.2, 4.4, 5.8, 1.6]
    #     ])
    
    # b = np.array([0.8, 3.1, -4.4, 2.6])

    A = np.array([
        [2.2, 4, -3, 1.5, 0.6, 2, 0.7],
        [4, 3.2, 1.5, -0.7, -0.8, 3, 1],
        [-3, 1.5, 1.8, 0.9, 3, 2, 2],
        [1.5, -0.7, 0.9, 2.2, 4, 3, 1],
        [0.6, -0.8, 3, 4, 3.2, 0.6, 0.7],
        [2, 3, 2, 3, 0.6, 2.2, 4],
        [0.7, 1, 2, 1, 0.7, 4, 3.2]
    ])

    b = np.array([3.2, 4.3, -0.1, 3.5, 5.3, 9.0, 3.7])

    # A = np.array([
    #     [2.2, 4, -3, 1.5, 0.6, 2, 0.7, 8.1, -4.5, 6.8],
    #     [4, 3.2, 1.5, -0.7, -0.8, 3, 1, 2.5, 6.3, 5.1],
    #     [-3, 1.5, 1.8, 0.9, 3, 2, 2, 0.5, 5.1, 4.3],
    #     [1.5, -0.7, 0.9, 2.2, 4, 3, 1, 5.3, -1, 6.4],
    #     [0.6, -0.8, 3, 4, 3.2, 0.6, 0.7, 5.1, 3.9, -1.4],
    #     [2, 3, 2, 3, 0.6, 2.2, 4, -5, 3.1, 2.5],
    #     [0.7, 1, 2, 1, 0.7, 4, 3.2, 5.1, 3.5, 4.4],
    #     [2.2, 4.3, -1.5, 0.8, 1.6, 1.1, -4.1, 1, 2, 3.4],
    #     [1.2, -4.5,  6.1, -2.3,  0.7,  3.4, -1.2,  4.8, -5.0,  2.9],
    #     [3.5,  2.1, -4.7,  1.8, -2.2,  4.0,  6.3, -0.9,  2.7, -1.4]
    # ])

    # b = np.array([3.2, 4.3, -0.1, 3.5, 5.3, 9.0, 3.7, 1.8, 4.1, -2])

    x = get_answer(A, b)

    print("\nВектор ответа (собственный метод):", x)

    norm_custom = np.linalg.norm(A @ x - b)
    print("\nПроверка (собственный метод): ||Ax - b|| =", norm_custom)

    # Проверка невязки
    residual = A @ x - b  # Невязка
    print("\nНевязка (Ax - b):", residual)


if __name__ == '__main__': 
    np.set_printoptions(linewidth=200, suppress=True)  
    main()