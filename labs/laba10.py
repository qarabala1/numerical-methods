import numpy as np

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

matrices = [
    np.array([[2.2, 1, 0.5, 2],
              [1, 1.3, 2, 1],
              [0.5, 2, 0.5, 1.6],
              [2, 1, 1.6, 2]]),
    
    np.array([[-0.168700, 0.353699, 0.008540, 0.733624],
              [0.353699, 0.056519, -0.723182, -0.076440],
              [0.008540, -0.723182, 0.015938, 0.342333],
              [0.733624, -0.076440, 0.342333, -0.045744]]),
    
    np.array([[1.00, 0.42, 0.54, 0.66],
              [0.42, 1.00, 0.32, 0.44],
              [0.54, 0.32, 1.00, 0.22],
              [0.66, 0.44, 0.22, 1.00]])
]

p = 8

for index, A in enumerate(matrices):
    print(f"Матрица {index + 1}:")
    
    solution, iterations = compute(A, p)

    eigenvalues = solution
    print(f"Завешилось за {iterations} итераций")
    print(f"Собственные значения: {eigenvalues}")
    for i in range(A.shape[0]):
        print(np.linalg.det(A - eigenvalues[i] * np.eye(A.shape[0])) )
    print()