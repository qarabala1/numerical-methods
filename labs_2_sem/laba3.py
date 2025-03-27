import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def phi(x, t):
    return (46 * np.pi * (9/4) - 6) * np.exp(-6 * t) * np.sin(6 * x / 4)

def psi(x):
    return np.sin(6 * x / 4)

def explicit_scheme(a, gamma_0, gamma_1, M, N, l, T):
    h = l / M
    tau = T / N
    lambda_ = (a**2 * tau) / h**2
    print(f"lambda (Явная схема) = {lambda_:.3f}")
    if lambda_ > 0.5:
        print("Предупреждение: Схема неустойчива!")

    x = np.linspace(0, l, M + 1)
    t = np.linspace(0, T, N + 1)
    u = np.zeros((N + 1, M + 1))

    # Начальное и граничные условия
    u[0, :] = psi(x)
    u[:, 0] = gamma_0
    u[:, -1] = gamma_1

    # Явная схема
    for n in range(N):
        for m in range(1, M):
            u[n+1, m] = u[n, m] + lambda_ * (u[n, m+1] - 2*u[n, m] + u[n, m-1]) + tau * phi(x[m], t[n])

    return x, t, u

def implicit_scheme(a, gamma_0, gamma_1, M, N, l, T):
    h = l / M
    tau = T / N
    lambda_ = (a**2 * tau) / h**2

    x = np.linspace(0, l, M + 1)
    t = np.linspace(0, T, N + 1)
    u = np.zeros((N + 1, M + 1))

    # Начальное и граничные условия
    u[0, :] = psi(x)
    u[:, 0] = gamma_0
    u[:, -1] = gamma_1

    # Построение матрицы для неявной схемы
    A = np.zeros((M-1, M-1))
    for i in range(M-1):
        if i > 0: A[i, i-1] = -lambda_
        A[i, i] = 1 + 2*lambda_
        if i < M-2: A[i, i+1] = -lambda_

    # Решение на каждом шаге
    for n in range(N):
        b = u[n, 1:-1].copy() + tau * phi(x[1:-1], t[n])
        u[n+1, 1:-1] = np.linalg.solve(A, b)

    return x, t, u

def exact_solution(x, t):
    return np.exp(-6 * t) * np.sin(6 * x / 4)

def plot_comparison(x, t, u_num, scheme_name):
    X, T = np.meshgrid(x, t)
    u_exact = exact_solution(X, T)
    error = u_exact - u_num

    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(scheme_name, fontsize=14)

    # Аналитическое решение
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X, T, u_exact, cmap='viridis', rstride=10, cstride=10)
    ax1.set_title("Аналитическое решение")

    # Численное решение
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X, T, u_num, cmap='viridis', rstride=10, cstride=10)
    ax2.set_title("Численное решение")

    # Погрешность
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(X, T, error, cmap='coolwarm', rstride=10, cstride=10)
    ax3.set_title("Погрешность")

    plt.tight_layout()
    plt.show()

def main():
    # Параметры задачи
    a = np.sqrt(46 * np.pi)  # a² = 46π
    gamma_0 = gamma_1 = 0    # Нулевые граничные условия
    l = (4 * np.pi) / 6      # Длина области
    M = 10                  # Узлов по пространству
    N = 20000                  # Узлов по времени (увеличено для устойчивости)
    T_max = 2              # Конечное время

    # Явная схема (неустойчива при данных параметрах!)
    x_exp, t_exp, u_exp = explicit_scheme(a, gamma_0, gamma_1, M, N, l, T_max)
    plot_comparison(x_exp, t_exp, u_exp, "Явная схема (Неустойчивая)")

    # Неявная схема
    x_imp, t_imp, u_imp = implicit_scheme(a, gamma_0, gamma_1, M, N, l, T_max)
    plot_comparison(x_imp, t_imp, u_imp, "Неявная схема")

if __name__ == "__main__":
    main()