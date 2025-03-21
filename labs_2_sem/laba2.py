import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2, suppress=True, linewidth=200, threshold=np.inf)

x_1 = 0.0
x_n = 1.0
N = 500 
h = (x_n - x_1) / N
x = np.linspace(x_1, x_n, N + 1)


def p(x):
    return -2  

def q(x):
    return 1    

def f(x):
    return x**2 - 4*x + 2  

def exact_solution(x):
    return np.exp(x) + x**2  

# def compute_a_c(r):
#     term = (2 - np.exp(r) - np.exp(-r))**2 / (2 + np.exp(r) + np.exp(-r))
#     a_i = (2 * np.abs(r) + term - r)/  h**2
#     c_i = (2 * np.abs(r) + term + r)/ h**2
#     return a_i, c_i

# def compute_a_c(r):
#     a_i = (np.abs(r) + np.exp(-np.abs(r)) - r) / h**2
#     c_i = (np.abs(r) + np.exp(-np.abs(r)) + r) / h**2
#     return a_i, c_i

def compute_a_c(r):
    a_i = (1 - r + r**2 / 2) / h**2
    c_i = (1 + r + r**2 / 2) / h**2
    return a_i, c_i

A = np.zeros((N + 1, N + 1))
d = np.zeros(N + 1)

alpha0, beta0, gamma0 = 1, 0, 1
A[0, 0] = alpha0
d[0] = gamma0

alpha1, beta1, gamma1 = 1, 0, np.exp(1) + 1
A[N, N] = alpha1
d[N] = gamma1

for i in range(1, N):
    r = (p(x[i]) / 2) * h  
    a_i, c_i = compute_a_c(r)
    
    A[i, i-1] = a_i  
    A[i, i] = - (a_i + c_i) + q(x[i])  
    A[i, i+1] = c_i  
    d[i] = f(x[i])

u_numerical = np.linalg.solve(A, d)

u_exact = exact_solution(x)

error = np.max(np.abs(u_numerical - u_exact))
print(error)

plt.figure(figsize=(12, 6))
plt.plot(x, u_numerical, 'b-', label='Численное решение')
plt.plot(x, u_exact, 'r--', label='Точное решение')
plt.title('Сравнение с модифицированной схемой')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
plt.grid(True)
plt.show()


