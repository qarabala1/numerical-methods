import numpy as np


def rotation_method(A: np.array, p: int = 10) -> np.array:
    def sgn(value):
        return 1 if value >= 0 else -1

    def compute(A: np.array, p: int):
        n = A.shape[0]
        
        iterations = 0
        for k in range(1, p + 1):
            sigma = [np.sqrt(max(abs(A[i, i]) for i in range(n))) / (10**p_i) for p_i in range(p + 1)]
            while True:
                max_abs_value = 0
                indexes = (0, 0)

                for i in range(n):
                    for j in range(i + 1, n):
                        if abs(A[i, j]) > max_abs_value:
                            max_abs_value = abs(A[i, j])
                            indexes = (i, j)

                if all(abs(A[i,j]) <= min(sigma) for i in range(n) for j in range(i + 1, n)):
                    break

                i, j = indexes

                # Объединенные функции для вычисления c и s
                d_val = np.sqrt((A[i, i] - A[j, j])**2 + 4 * A[i, j]**2)
                c_val = np.sqrt(0.5 * (1 + abs(A[i, i] - A[j, j]) / d_val))
                s_val = sgn(A[i, j] * (A[i, i] - A[j, j])) * np.sqrt(0.5 * (1 - abs(A[i, i] - A[j, j]) / d_val))

                C = A.copy()
                for k in range(n):
                    if k != i and k != j:
                        C[k, i] = c_val * A[k, i] + s_val * A[k, j]
                        C[i, k] = C[k, i]
                        C[k, j] = -s_val * A[k, i] + c_val * A[k, j]
                        C[j, k] = C[k, j]

                C[i, i] = c_val**2 * A[i, i] + 2 * c_val * s_val * A[i, j] + s_val**2 * A[j, j]
                C[j, j] = s_val**2 * A[i, i] - 2 * c_val * s_val * A[i, j] + c_val**2 * A[j, j]
                C[i, j] = 0
                C[j, i] = 0

                A = C
                iterations += 1

        eigenvalues = sorted(A[i, i] for i in range(n))
        return eigenvalues, iterations

    return compute(A, p)


def checkMatrix(matrix):
    if matrix.shape[0] != matrix.shape[1]:
        return False

    if not np.allclose(matrix, matrix.T):
        return False

    minors = [np.linalg.det(matrix[:i, :i]) for i in range(1, matrix.shape[0] + 1)]
    return all(minor > 0 for minor in minors)


def richardson_method(A: np.array, b: np.array, x: np.array, eps: float = 1e-9, max_iter=100, n=20):
    if not checkMatrix(A):
        print("Матрица не положительно определена и/или не симметрическая")
        return x, 0

    lambdas, i = rotation_method(A)
    lambda_min, lambda_max = lambdas[0], lambdas[-1]
    tau_0 = 2 / (lambda_min + lambda_max)
    rho_0 = (1 - lambda_min / lambda_max) / (1 + lambda_min / lambda_max)

    iterations = 0

    for iteration in range(max_iter):
        if np.linalg.norm(A @ x - b) <= eps:
            break
        for k in range(1, n + 1):
            v_k = np.cos((2 * k - 1) * np.pi / (2 * n))
            t_k = tau_0 / (1 + rho_0 * v_k)
            x = (b - A @ x) * t_k + x
        iterations += 1

    return x, iterations * n


def check_solution(A: np.array, b: np.array, x: np.array):
    # Получение истинного решения через встроенный метод
    true_solution = np.linalg.solve(A, b)

    print("Решение с помощью встроенного методода:", true_solution)
    print("A @ x - b:", A @ x - b)
    print()


def check_eigenvalues(A: np.array):
    # Вычисление собственных значений через метод вращения
    computed_eigenvalues, i = rotation_method(A)

    # Вычисление собственных значений через встроенный метод
    true_eigenvalues = np.linalg.eigvals(A)
    print("Выполнен поиск собственных значений. Потребовалось итераций: ", i)
    print("Собственные значения с помощью встроенных методов:", np.sort(true_eigenvalues))
    print("Собственные значения, полученные методом вращения:", computed_eigenvalues)

    # Проверка характеристического уравнения
    dets = [np.linalg.det(A - eigenval * np.eye(A.shape[0])) for eigenval in computed_eigenvalues]
    print("Определители для det(A - I*lambda):", dets)
    print()


def main():
    matrices = {
        "A": (np.array(
            [[2, 1],
             [1, 2]]),
             np.array([4, 5])),
    }

    for name, (A, b) in matrices.items():
        print(name)

        x = np.zeros_like(b)
        x, iterations = richardson_method(A, b, x)

        print(f"Итерационный процесс завершился за {iterations} итераций с решением: {x} \n")

        check_solution(A, b, x)


        # Проверка собственных значений
        check_eigenvalues(A)


if __name__ == '__main__':
    main()

