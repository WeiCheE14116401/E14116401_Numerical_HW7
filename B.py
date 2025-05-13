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
tolerance = 1e-8    #小數上限

def gauss_seidel_method(A, b, x_initial, tol=1e-8, max_iter=1000):
    n = len(b)
    x = x_initial.copy()
    
    print("\nGauss-Seidel :")
    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            sigma1 = 0
            for j in range(i): # j < i
                sigma1 += A[i, j] * x[j] # 使用本輪已更新的 x[j]
            sigma2 = 0
            for j in range(i + 1, n): # j > i
                sigma2 += A[i, j] * x_old[j] # 使用上一輪的 x_old[j]
            x[i] = (b[i] - sigma1 - sigma2) / A[i, i]
        print(f"  @Times= {k+1}: {x}") #每一次的計算結果
        if np.linalg.norm(x - x_old, np.inf) < tol:
            print(f"  converge in {k+1} iterations.") #在幾次內收斂
            return x

    print("  Maybe not convergence.")
    return x

x_gauss_seidel = gauss_seidel_method(A, b, x_initial.copy(), tolerance, max_iterations)
x_gauss_seidel = np.round(x_gauss_seidel, 5)
print(f"Solution by Gauss-Seidel: {x_gauss_seidel}")
