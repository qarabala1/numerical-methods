import matplotlib.pyplot as plt
import numpy as np

def solve_hyperbolic_eq(a, l, T, h, tau, phi, psi, mu1, mu2, f=lambda x, t: 0):
    # Проверка условия Куранта
    cfl = a * tau / h
    if cfl > 1:
        raise ValueError(f"Условие Куранта нарушено (CFL={cfl:.2f} > 1). Уменьши tau или увеличь h.")

    # Создаем сетки (x — первая ось, t — вторая)
    x = np.linspace(0, l, int(l/h) + 1)
    t = np.linspace(0, T, int(T/tau) + 1)
    u = np.zeros((len(x), len(t)))  # Форма (Nx, Nt)

    # Начальное условие u(x, 0) = phi(x)
    u[:, 0] = phi(x)  # Весь первый временной слой (t=0)

    # Первый временной слой (t=1)
    u[1:-1, 1] = (
        u[1:-1, 0]  # Значения с предыдущего времени (t=0)
        + tau * psi(x[1:-1])  # Начальная скорость
        + (a**2 * tau**2 / (2*h**2)) * (u[2:, 0] - 2*u[1:-1, 0] + u[:-2, 0])  # Пространственная производная
        + (tau**2 / 2) * f(x[1:-1], 0)  # Внешняя сила
    )

    # Граничные условия для t=1
    u[0, 1] = mu1(t[1])   # Левый конец
    u[-1, 1] = mu2(t[1])  # Правый конец

    # Основной цикл (t=2, 3, ...)
    for n in range(1, len(t)-1):
        # Внутренние точки (x=1, 2, ..., Nx-2)
        u[1:-1, n+1] = (
            2 * u[1:-1, n] - u[1:-1, n-1]  # Волновое уравнение
            + (a**2 * tau**2 / h**2) * (u[2:, n] - 2*u[1:-1, n] + u[:-2, n])  # Пространственная производная
            + tau**2 * f(x[1:-1], t[n])  # Внешняя сила
        )
        # Граничные условия
        u[0, n+1] = mu1(t[n+1])
        u[-1, n+1] = mu2(t[n+1])

    return x, t, u

# Параметры
a = 1.0
l = 1.0
T = 10.0
h = 0.02
tau = 0.02

# Начальные и граничные условия
phi = lambda x: 3 * x * (1 - x)
psi = lambda x: -2 * (x**2 - x)
mu1 = lambda t: 0
mu2 = lambda t: 0
f = lambda x, t: t * x**2 * (x - 1)

# Решение
x, t, u = solve_hyperbolic_eq(a, l, T, h, tau, phi, psi, mu1, mu2, f)

# Визуализация (исправлено!)
plt.figure(figsize=(10, 6))
cmap = plt.cm.viridis

for ts in np.arange(0, 1.1, 0.1):
    # Находим индекс временного слоя
    n = np.argmin(np.abs(t - ts))
    # Берем ВСЕ точки по x для времени n: u[:, n]
    plt.plot(x, u[:, n], color=cmap(ts / 1.1), lw=1.5, alpha=0.7, label=f't={ts:.1f}')

plt.title('Эволюция струны')
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.grid(alpha=0.2)
plt.legend(ncol=3, title='Время →')
plt.tight_layout()
plt.show()