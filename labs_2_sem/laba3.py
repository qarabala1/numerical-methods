import numpy as np
import matplotlib.pyplot as plt


a_sq = 16 / np.pi
L = 4.0          
T_max = 1.0      
Nx = 100         
Nt = 20000

h = L / Nx       
tau = T_max / Nt 

x = np.linspace(0, L, Nx + 1)  
t = np.linspace(0, T_max, Nt + 1)  
X, T = np.meshgrid(x, t)

def analytical_solution(x, t):
    return np.exp(-np.pi * t) * np.sin(np.pi * x / 4)

U_analytical = analytical_solution(X, T)

u_explicit = np.zeros((Nx + 1, Nt + 1)) 
u_explicit[:, 0] = analytical_solution(x, 0)
gamma = a_sq * tau / h**2  

for n in range(Nt):  
    u_explicit[1:-1, n+1] = u_explicit[1:-1, n] + gamma * (
        u_explicit[2:, n] - 2*u_explicit[1:-1, n] + u_explicit[:-2, n]
    )
    u_explicit[0, n+1] = 0
    u_explicit[-1, n+1] = 0

u_implicit = np.zeros((Nx + 1, Nt + 1))  
u_implicit[:, 0] = analytical_solution(x, 0)
alpha = gamma

for n in range(Nt):
    d = u_implicit[1:-1, n].copy()
    size = len(d) 
    
    a = -alpha * np.ones(size - 1)  
    b = (1 + 2*alpha) * np.ones(size)  
    c = -alpha * np.ones(size - 1)  
    
    for i in range(1, size):
        m = a[i-1] / b[i-1]
        b[i] -= m * c[i-1]
        d[i] -= m * d[i-1]
    
    u_new = np.zeros(size)
    u_new[-1] = d[-1] / b[-1]
    for i in range(size-2, -1, -1):
        u_new[i] = (d[i] - c[i] * u_new[i+1]) / b[i]
    
    u_implicit[1:-1, n+1] = u_new
    u_implicit[0, n+1] = 0
    u_implicit[-1, n+1] = 0

error_explicit = np.abs(u_explicit.T - U_analytical)
error_implicit = np.abs(u_implicit.T - U_analytical)
print('Максимальная погрешность явной схемы:', np.max(error_explicit))
print('Максимальная погрешность неявной схемы', np.max(error_implicit))
print('tau =', tau)



fig = plt.figure(figsize=(18, 12))

alpha_value = 0.7

# Явная схема
ax1 = fig.add_subplot(231, projection='3d')
ax1.plot_surface(X, T, u_explicit.T, cmap='viridis', alpha=alpha_value)
ax1.set_title('Явная схема')
ax1.set_xlabel('x')
ax1.set_ylabel('t')

# Неявная схема
ax2 = fig.add_subplot(232, projection='3d')
ax2.plot_surface(X, T, u_implicit.T, cmap='viridis', alpha=alpha_value)
ax2.set_title('Неявная схема')
ax2.set_xlabel('x')
ax2.set_ylabel('t')

# Аналитическое решение
ax3 = fig.add_subplot(233, projection='3d')
ax3.plot_surface(X, T, U_analytical, cmap='viridis', alpha=alpha_value)
ax3.set_title('Аналитическое решение')
ax3.set_xlabel('x')
ax3.set_ylabel('t')

# Погрешность явной схемы
ax4 = fig.add_subplot(234, projection='3d')
ax4.plot_surface(X, T, error_explicit, cmap='hot', alpha=alpha_value)
ax4.set_title('Погрешность явной схемы')
ax4.set_xlabel('x')
ax4.set_ylabel('t')

# Погрешность неявной схемы
ax5 = fig.add_subplot(235, projection='3d')
ax5.plot_surface(X, T, error_implicit, cmap='hot', alpha=alpha_value)
ax5.set_title('Погрешность неявной схемы')
ax5.set_xlabel('x')
ax5.set_ylabel('t')

plt.tight_layout()
plt.show()