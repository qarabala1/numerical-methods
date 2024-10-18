import numpy as np

def gauss_with_partial_pivoting(A, b):
    n = len(A)
    M = np.hstack((A, b.reshape(-1, 1)))  
    
    # Прямой ход
    for k in range(n):
        submatrix = M[k:n, k:n]  
        max_row = np.argmax(np.abs(submatrix[:, 0])) + k  

        if np.abs(M[max_row, k]) == 0:
            raise ValueError("Система не имеет единственного решения.")
        
        if max_row != k:
            M[[k, max_row]] = M[[max_row, k]]
        
        for i in range(k + 1, n):
            m = M[i, k] / M[k, k]
            M[i, k:] -= m * M[k, k:]
    
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (M[i, -1] - np.dot(M[i, i + 1:n], x[i + 1:])) / M[i, i]
    
    return x

# Пример использования
# A = np.array([[2.0, -1.0, 1.0], 
#               [1.0, 3.0, -2.0], 
#               [3.0, 1.0, 2.0]])  # матрица коэффициентов

# b = np.array([7.0, -3.0, 12.0])  # вектор свободных членов

def input_matrix_and_vector():
    print("Введите матрицу коэффициентов A в формате [[...], [...], ...]:")
    A = eval(input("A = "))
    
    print("Введите вектор свободных членов b в формате [...]:")
    b = eval(input("b = "))

    return np.array(A), np.array(b)

# Ввод данных и вычисление
A, b = input_matrix_and_vector()
result = gauss_with_partial_pivoting(A, b)
print("Решение системы:", result)

