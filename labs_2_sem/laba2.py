import numpy as np
import matplotlib.pyplot as plt

x_1 = 0.0
x_n = 1.0
N = 100
h = (x_n - x_1) / N
x = np.linspace(x_1, x_n, N + 1)

def p(x):
    return -(x + 1.0)**2

def q(x):
    return -2.0 / (x + 1.0)**2

def f_func(x):
    return 1.0

def exact_solution(x):
    return 1.0 / (x + 1.0)

def compute_a_c(r):
    term = np.tan(np.abs(r)) + 1.0 / (np.abs(r) + 1.0)
    a_i = (term - r) / h**2
    c_i = (term + r) / h**2
    return a_i, c_i

A = np.zeros((N + 1, N + 1))
d = np.zeros(N + 1)

A[0, 0] = 1.0 + 3.0/(2.0*h)
A[0, 1] = -4.0/(2.0*h)
A[0, 2] = 1.0/(2.0*h)
d[0] = 2.0

A[N, N] = 1.0
d[N] = 0.5

for i in range(1, N):
    r_i = 0.5 * p(x[i]) * h
    a_i, c_i = compute_a_c(r_i)
    
    if i == 1:
        coeff_u0 = A[0, 0]  
        coeff_u1 = A[0, 1]  
        coeff_u2 = A[0, 2] 

        A[1, 1] += A[1, 0] * (-coeff_u1 / coeff_u0)
        A[1, 2] += A[1, 0] * (-coeff_u2 / coeff_u0)
        d[1] += A[1, 0] * (d[0] / coeff_u0)
        A[1, 0] = 0.0  # a1 = 0
        
    A[i, i-1] = a_i
    A[i, i] = - (a_i + c_i) + q(x[i])
    A[i, i+1] = c_i
    d[i] = f_func(x[i])

u_numerical = np.linalg.solve(A, d)

u_exact = exact_solution(x)
error = np.max(np.abs(u_numerical - u_exact))
print("Максимальная погрешность =", error)

plt.figure(figsize=(10, 5))
plt.plot(x, u_numerical, 'b.-', label='Численное решение')
plt.plot(x, u_exact, 'r--', label='Точное решение')
plt.title(f'Сравнение решений, погрешность: {error:.3e}')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
plt.grid(True)
plt.show()