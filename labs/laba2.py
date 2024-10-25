import numpy as np

def gauss_with_full_pivoting(A, b):
    n = len(b)
    M = np.hstack([A, b.reshape(-1, 1)])  # Объединяем A и b
    
    # Массив для отслеживания порядка переменных
    index_permutation = np.arange(n)
    print(M)
    for i in range(n):
        print(f'M{i+1} = ')
        # Находим индекс главного элемента
        max_row_index, max_col_index = np.unravel_index(np.abs(M[i:n, i:n]).argmax(), M[i:n, i:n].shape)
        max_row_index += i  # Приводим индексы к полной матрице
        max_col_index += i
        
        # Меняем местами строки
        M[[i, max_row_index]] = M[[max_row_index, i]]
        # Меняем местами столбцы
        M[:, [i, max_col_index]] = M[:, [max_col_index, i]]
        
        # Обновляем порядок переменных
        index_permutation[[i, max_col_index]] = index_permutation[[max_col_index, i]]
        
        # Приводим матрицу к верхнетреугольному виду
        for j in range(i + 1, n):
            m = M[j, i] / M[i, i]
            M[j, i:] -= m * M[i, i:]

        print(M)
    # Обратная подстановка
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (M[i, -1] - np.dot(M[i, i + 1:n], x[i + 1:n])) / M[i, i]
    
    ordered_x = np.empty_like(x)
    ordered_x[index_permutation] = x
    
    return ordered_x

# Пример использования
A = np.array([[1, 5, -4, 1, 0, 8, 0],
              [2, 3, 3, 1, 6, 10, 11],
              [1, -3, 1, 2, 7, 6, 14],
              [4, 2, 1, -1, 1, 12, 6], [1, 2, 3, 4, 0, 11, 8], [11, 7, 0, -11, 8, 9, 4], [9, 8, 1, -11, 4, 9, 6]], dtype=float)
b = np.array([3, 13, 12, 6, 5, -1, -2], dtype=float)

solution = gauss_with_full_pivoting(A, b)
print("Решение системы:", solution)

# Сравнение с numpy
numpy_solution = np.linalg.solve(A, b)
print("Решение с использованием numpy.linalg.solve:", numpy_solution)

# Проверка разности
difference = solution - numpy_solution
print("Разница между решениями:", difference)
