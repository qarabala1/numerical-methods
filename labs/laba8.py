import numpy as np


def simple_iteration(A, b, x0=None, eps=1e-10, max_iter=1000):
    n = len(b)
    if x0 is None:
        x0 = np.zeros(n)

    x = np.copy(x0)

    for iteration in range(max_iter):
        # Вычисление нового приближения
        x_new = (b - np.dot(A, x) + np.diag(A) * x) / np.diag(A)

        if np.linalg.norm(x_new - x) < eps:
            return x_new, iteration

        x = x_new

    print(f"Метод простой итерации не сошелся за {max_iter} итераций.")
    return x, max_iter


def relaxation_method(A, b, omega, x0=None, eps=1e-6, max_iter=10000):
    n = A.shape[0]
    if x0 is None:
        x0 = np.zeros(n)

    x = x0.copy()

    for k in range(max_iter):
        x_new = x.copy()

        for i in range(n):
            sum1 = np.dot(A[i, :i], x_new[:i])
            sum2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (1 - omega) * x[i] + (omega / A[i, i]) * (b[i] - sum1 - sum2)

        # Проверка на сходимость
        if np.linalg.norm(x_new - x) < eps:
            return x_new, k

        x = x_new

    print(f"Метод релаксации не сошелся за {max_iter} итераций.")
    return x, max_iter


if __name__ == '__main__':
    A = np.array([
        [4, 1, 1, 1, 1],
        [2, 9, 3, 2, 2],
        [1, 2, 7, 1, 2],
        [1, 1, 1, 7, 3],
        [2, 3, 4, 1 , 11]
    ])

    b = np.array([10, 12, 14, 16, 18])


    omegas = [0.1, 0.5, 1, 1.5, 1.7, 1.9]

    for omega in omegas:
        x, iterations = relaxation_method(A, b, omega)
        print(f"omega = {omega}: Решение = {x}, Количество итераций = {iterations}\n")
    x, iterations = simple_iteration(A, b)
    print(f"Решение методом простой итерации: {x} Количество итераций: {iterations}\n")

    x_numpy = np.linalg.solve(A, b)
    print("Решение с использованием np.linalg.solve:", x_numpy)