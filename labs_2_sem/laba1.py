import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return y - x**2 + 1

def exact_solution(x):
    return (x + 1)**2 - 0.5 * np.exp(x)

def euler_method(f, x0, y0, h, n):
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    x[0], y[0] = x0, y0
    for i in range(n):
        y[i + 1] = y[i] + h * f(x[i], y[i])
        x[i + 1] = x[i] + h
    return x, y

def improved_euler_method(f, x0, y0, h, n):
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    x[0], y[0] = x0, y0
    for i in range(n):
        y_pred = y[i] + h * f(x[i], y[i])
        x_next = x[i] + h
        y_avg = y[i] + h * (f(x[i], y[i]) + f(x_next, y_pred)) / 2
        x[i + 1], y[i + 1] = x_next, y_avg
    return x, y

def runge_kutta_4th_order(f, x0, y0, h, n):
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    x[0], y[0] = x0, y0
    for i in range(n):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + h/2, y[i] + (h/2)*k1)
        k3 = f(x[i] + h/2, y[i] + (h/2)*k2)
        k4 = f(x[i] + h, y[i] + h*k3)
        y[i + 1] = y[i] + h*(k1 + 2*k2 + 2*k3 + k4)/6
        x[i + 1] = x[i] + h
    return x, y

def plot_solutions(x_exact, y_exact, x_numerical, y_numerical, labels, title):
    plt.figure(figsize=(10, 6))
    plt.plot(x_exact, y_exact, 'k-', label='Точное решение')
    for i in range(len(x_numerical)):
        plt.plot(x_numerical[i], y_numerical[i], '--', label=labels[i])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

x0, y0 = 0.0, 0.5  
n = 500
x_end = 2  
h = (x_end - x0) / n  

x_euler, y_euler = euler_method(f, x0, y0, h, n)
x_improved, y_improved = improved_euler_method(f, x0, y0, h, n)
x_rk4, y_rk4 = runge_kutta_4th_order(f, x0, y0, h, n)

y_exact_euler = exact_solution(x_euler)
y_exact_improved = exact_solution(x_improved)
y_exact_rk4 = exact_solution(x_rk4)

max_error_euler = np.max(np.abs(y_euler - y_exact_euler))
max_error_improved = np.max(np.abs(y_improved - y_exact_improved))
max_error_rk4 = np.max(np.abs(y_rk4 - y_exact_rk4))

print(f"Максимальная погрешность:")
print(f"- Метод Эйлера: {max_error_euler:.6e}")
print(f"- Улучшенный Эйлер: {max_error_improved:.6e}")
print(f"- Рунге-Кутта: {max_error_rk4:.6e}")

x_exact = np.linspace(x0, x_end, 100)
y_exact = exact_solution(x_exact)

plot_solutions(x_exact, y_exact, 
               [x_euler, x_improved, x_rk4], 
               [y_euler, y_improved, y_rk4], 
               ['Метод Эйлера', 'Улучшенный Эйлер', 'Рунге-Кутта'], 
               'Сравнение методов решения y\' = y - x^2 + 1')