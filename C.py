import numpy as np
A = np.array([
    [4, -1,  0, -1,  0,  0],
    [-1, 4, -1,  0, -1,  0],
    [0, -1,  4,  0,  1, -1],
    [-1, 0,  0,  4, -1, -1],
    [0, -1,  0, -1,  4, -1],
    [0,  0, -1,  0, -1,  4]
], dtype=float)

b = np.array([0, -1, 9, 4, 8, 6], dtype=float)

# 猜測值設定
x_initial = np.zeros_like(b)
# 收斂條件
max_iterations = 1000 #上限次數
tolerance = 1e-8 #小數上限

def sor_method(A, b, x_initial, omega, tol=1e-8, max_iter=1000):
    n = len(b)
    x = x_initial.copy()
    
        
    print(f"\nSOR method (omega = {omega}):")
    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            sigma1 = 0
            for j in range(i): # j < i
                sigma1 += A[i, j] * x[j] # 使用本輪已更新的 x[j]
            sigma2 = 0
            for j in range(i + 1, n): # j > i
                sigma2 += A[i, j] * x_old[j] # 使用上一輪的 x_old[j]
            
            x_gauss_seidel_i = (b[i] - sigma1 - sigma2) / A[i, i]
            x[i] = (1 - omega) * x_old[i] + omega * x_gauss_seidel_i
            
        if np.linalg.norm(x - x_old, np.inf) < tol:
            print(f"  converge in {k+1} iterations.") #在幾次內收斂
            return x
        
        print(f"  @Times={k+1}: {x}") #每一次的計算結果


    print("  Maybe not convergence.")
    return x

omega_sor = 1.1 
x_sor = sor_method(A, b, x_initial.copy(), omega_sor, tolerance, max_iterations)
x_sor = np.round(x_sor, 5)
print(f"solution by SOR method (omega={omega_sor}): {x_sor}")