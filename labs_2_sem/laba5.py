import numpy as np 
import sympy as sp
from prettytable import PrettyTable
import matplotlib.pyplot as plt

def exact_solution(x):
    return 1.0 / (x + 1.0)

def phi(x, k):
    if k == 0:
        return 0.5 * x
    return (1 - x) ** k 

def p(x):
    return np.exp(-(x + 1.0)**2)

def q(x):
    return -2.0 / (x + 1.0)**2

def f_func(x):
    return 1.0

def J_1(u, x, p_func, q_func, f_func):
    return p_func(x) * (sp.diff(u, x) ** 2) - q_func(x) * (u ** 2) + 2 * f_func(x) * u

def build_u(x, c_list):
    """Формируем линейную комбинацию базисных функций."""
    return sum(c_list[k - 1] * phi(x, k) for k in range(1, len(c_list) + 1)) + phi(x, 0)

def dPhi_dCi(i, u_expr, x, c_list, a, b, p_func, q_func, f_func):
    """Дифференциал функционала J по i-й переменной."""
    return sp.integrate(
        sp.diff(J_1(u_expr, x, p_func, q_func, f_func), c_list[i]),
        (x, a, b)
    )

def main():
    n = 5  # Количество базисных функций, включая phi_0
    a, b = 0, 1
    x = sp.symbols('x')
    c_list = sp.symbols(f'c1:{n}')  # создаёт кортеж (c1, c2, ..., c_{n-1})

    A = np.zeros((n - 1, n - 1))
    b_vec = np.zeros(n - 1)

    u_expr = build_u(x, c_list)

    for i in range(n - 1):
        expr = dPhi_dCi(i, u_expr, x, c_list, a, b, p, q, f_func)
        for j in range(n - 1):
            A[i][j] = expr.coeff(c_list[j])
        b_vec[i] = -expr.subs({c: 0 for c in c_list})

    solution = np.linalg.solve(A, b_vec)

    u_final = u_expr
    for i in range(n - 1):
        u_final = u_final.subs(c_list[i], solution[i])

    ab = np.linspace(a, b, 10)
    my_values = [u_final.subs(x, xi).evalf() for xi in ab]
    true_values = [exact_solution(xi) for xi in ab]
    errors = [abs(true_values[i] - my_values[i]) for i in range(len(ab))]

    table = PrettyTable()
    table.field_names = ["x", "Истинное значение", "Ритц", "Ошибка"]
    mse = sum((float(true_values[i]) - float(my_values[i]))**2 for i in range(len(ab))) / len(ab)
    rmse = np.sqrt(mse)
    print(f"Среднеквадратичная ошибка (RMSE): {rmse:.12f}")


    plt.figure(figsize=(10, 6))
    plt.plot(ab, true_values, label="Истинное решение", color="blue", linestyle='--')
    plt.plot(ab, my_values, label="Ритц решение", color="red", linestyle='-')
    plt.plot(ab, errors, label="Ошибка", color="green", linestyle=':')
    plt.title('Сравнение решений')
    plt.xlabel('x')
    plt.ylabel('Значения')
    plt.legend()
    plt.grid(True)  
    plt.show()

if __name__ == '__main__':
    main()
