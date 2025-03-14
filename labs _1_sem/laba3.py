import numpy as np

# Функция для печати матрицы
def print_matrix(matrix):
    for row in matrix:
        print(" ".join(map(str, row)))
    print()

# Функция для выполнения базовой прямой и обратной подстановки
def basic_elimination(M: np.ndarray) -> np.ndarray:
    for i in range(M.shape[0]):
        M = M.astype(float)
        for j in range(i):
            M[i, :] -= M[i, j] * M[j, :]  # Обнуляем элементы в строке

        M[i, :] /= M[i, i]  # Нормируем строку

        for j in range(i):
            M[j, :] -= M[j, i] * M[i, :]  # Обнуляем верхние элементы

        print_matrix(M)  

    return M


# Исходная матрица
M = np.array(
    [
        [5, 2, 3, 3, 5, 8],
        [1, 6, 1, 5, -11, 12],
        [3, -4, -2, 8, 14, 15],
        [1, 9, 10, 0, -11, 13],
        [4, -7, 8, 19, 11, 7]
    ]
)

# Выполняем исключение и получаем матрицу с решением
result_matrix = basic_elimination(M)
solution = result_matrix[:, -1]  # Последний столбец содержит решения

A = M[:, :-1] 
B = M[:, -1]  
Ax = A @ solution 

print("Матрица")
print("Решение:", solution)
print("A * X:", Ax)
print("B:", B)
print("Разница:", abs(Ax - B))
