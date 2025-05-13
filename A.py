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

def jacobi_method(A, b, x_initial, tol=1e-8, max_iter=1000):
    n = len(b)
    x = x_initial.copy()
    x_new = np.zeros_like(x)
    
    print("\nJacobi method:")
    for k in range(max_iter):
        for i in range(n):
            sigma = 0
            for j in range(n):
                if i != j:
                    sigma += A[i, j] * x[j]
            x_new[i] = (b[i] - sigma) / A[i, i]
        
        if np.linalg.norm(x_new - x, np.inf) < tol:
            print(f"  converge in {k+1} iterations.") #在幾次內收斂
            return x_new
        x = x_new.copy()  #每一次的計算結果
        print(f"  @Times={k+1}: x = {x}")
    
    print("  Maybe not convergence.")
    return x

x_jacobi = jacobi_method(A, b, x_initial.copy(), tolerance, max_iterations)
x_jacobi = np.round(x_jacobi, 5)
print(f"Solution by jacobi method: {x_jacobi}")
