import numpy as np

def conjugate_gradient(A, b, x_init, tol=1e-8, max_iter=1000):

    x = np.array(x_init, dtype=float)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    for k in range(max_iter):

        v = b - (A @ x) 
        Ar = A @ v 
        rTr = v @ v  
        rTAr = v @ Ar 

        alpha = rTr / rTAr
        # 更新解向量 x(n+1) = x(n) + alpha * v(n)

        x_new = x + alpha * v
        # 計算誤差
        error = np.linalg.norm(x_new - x)
        x = x_new  
        #更新x

        if error < tol:
            break
    else: # not converge
        print(f"\nMaybe not convergence.")
        
    return x

if __name__ == "__main__":
    n = 6

    # 初始化矩陣 A
    A_matrix = np.array([
    [4, -1,  0, -1,  0,  0],
    [-1, 4, -1,  0, -1,  0],
    [0, -1,  4,  0,  1, -1],
    [-1, 0,  0,  4, -1, -1],
    [0, -1,  0, -1,  4, -1],
    [0,  0, -1,  0, -1,  4]
    ], dtype=float)

    b_vector = np.array([0, -1, 9, 4, 8, 6], dtype=float)
    
    # 猜測值設定
    x_initial = np.zeros(n)

    # 收斂條件
    max_iterations = 1000 #上限次數
    tolerance = 1e-8 #小數上限

    solution_x = conjugate_gradient(A_matrix, b_vector, x_initial, 
                                        tol=tolerance, max_iter=max_iterations)

    # 輸出結果
    print("\n最終解:")
    for i in range(n):
        print(f"x[{i + 1}] = {solution_x[i]:.5f}")
