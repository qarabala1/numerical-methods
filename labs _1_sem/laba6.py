import numpy as np

# Основная функция нахождения обратной матрицы
def matrix_inverse(A):
    n = A.shape[0]  # Размер матрицы
    A_inv = np.zeros_like(A, dtype=float)  # Создаем пустую матрицу для хранения обратной

    # Базовый случай: если матрица 1x1
    A_inv[0, 0] = 1 / A[0, 0]

    # Процесс окаймления
    for k in range(1, n):
        # Подматрица A_k
        A_k = A[:k, :k]
        
        # Вектора u и v
        u = A[:k, k]  # столбец u
        v = A[k, :k]  # строка v^T
        a_kk = A[k, k]  # элемент a_{k,k}

        # Находим обратную для подматрицы A_k
        A_k_inv = A_inv[:k, :k]

        # Величина для корректировки
        denom = a_kk - v @ A_k_inv @ u

        # Формируем новые блоки обратной матрицы
        A_inv_new = np.zeros((k + 1, k + 1))
        A_inv_new[:k, :k] = A_k_inv + (A_k_inv @ u[:, None] @ v[None, :] @ A_k_inv) / denom
        A_inv_new[:k, k] = -A_k_inv @ u / denom
        A_inv_new[k, :k] = -v @ A_k_inv / denom
        A_inv_new[k, k] = 1 / denom

        # Обновляем текущую обратную матрицу
        A_inv[:k + 1, :k + 1] = A_inv_new

    return A_inv


def main():
    A = np.array([
        [0.411, 0.421, -0.333, 0.313, -0.141, -0.381, 0.245],
        [0.241, 0.705, 0.139, -0.409, 0.321, 0.0625, 0.101],
        [0.123, -0.239, 0.502, 0.901, 0.243, 0.819, 0.321],
        [0.413, 0.309, 0.801, 0.865, 0.423, 0.118, 0.183],
        [0.241, -0.221, -0.243, 0.134, 1.274, 0.712, 0.423],
        [0.281, 0.525, 0.719, 0.118, -0.974, 0.808, 0.923],
        [0.246, -0.301, 0.231, 0.813, -0.702, 1.223, 1.105],
    ])
    b = np.array([0.096, 1.252, 1.024, 1.023, 1.155, 1.937, 1.673])

    A_inv = matrix_inverse(A)

    # Печать обратной матрицы
    print("\nОбратная матрица:\n", A_inv)

    # Проверка: произведение A и A_inv должно быть единичной матрицей
    identity_check = A @ A_inv
    print("\nПроверка: A * A_inv =\n", identity_check)

    # Вывод матрицы разниц (A * A_inv - единичная матрица)
    identity_matrix = np.eye(A.shape[0])  # Создание единичной матрицы
    difference_matrix = identity_check - identity_matrix
    print("\nМатрица разниц (A * A_inv - единичная матрица):\n", difference_matrix)

    # Проверка, что диагональные элементы обратной матрицы равны единице
    diagonal_elements = np.diag(identity_check)
    if np.allclose(diagonal_elements, 1):
        print("Все диагональные элементы равны единице.")
    else:
        print("Некоторые диагональные элементы не равны единице:", diagonal_elements)

    # Решение системы уравнений Ax = b
    x = A_inv @ b  # Умножение обратной матрицы на вектор b
    print("\nРешение системы Ax = b:\n", x)

    print(A @ x - b)
if __name__ == '__main__':
    main()
