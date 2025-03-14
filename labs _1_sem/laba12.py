import numpy as np

def simple_iter(A: np.ndarray, x0: np.ndarray = None,eps: float = 1e-1, max_iter: int = 10**5,):
    n = A.shape[0]

    if x0 is not None:
        x = x0 / np.linalg.norm(x0)
    else:
        x = np.random.rand(n)
        x /= np.linalg.norm(x)

    l = 0  
    for iter in range(max_iter):
        y = A @ x  
        l_new = y @ x  

        y_norm = y / np.linalg.norm(y)  

        if np.abs(l - l_new) < eps:
            print(f"Итераций: {iter}, разница: {np.abs(l - l_new)}")
            return x, l_new, iter 

        x = y_norm 
        l = l_new  

    return x, l, iter  


A = np.array(
    [
        [-0.168700, 0.353699, 0.008540, 0.733624],
        [0.353699, 0.056519, -0.723182, -0.076440],
        [0.008540, -0.723182, 0.015938, 0.342333],
        [0.733624, -0.076440, 0.342333, -0.045744],
    ]
)

eigvec, eigval, iters = simple_iter(A, eps=1e-2, max_iter=10**5)

np_mx_eigval = np.max(np.abs(np.linalg.eigvals(A))) 
print(f"Собственное значение {eigval}")
print(f"Разница встроенного метода и метода простой итерации: {np.abs(np_mx_eigval - np.abs(eigval))}\n")
print(f"Собственный вектор: {eigvec}")
print((A @ eigvec) / eigval - eigvec)
