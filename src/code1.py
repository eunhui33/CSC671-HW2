import random
m_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

def generate_matrices(m):
    W = [[random.random() for _ in range(m)] for _ in range(90)]
    X = [[random.random() for _ in range(110)] for _ in range(m)]
    return W, X



def plain_matmul(W, X, m):
    rows_W = len(W)      # 90
    cols_X = len(X[0])   # 110
    result = [[0.0] * cols_X for _ in range(rows_W)]
    for i in range(rows_W):
        for j in range(cols_X):
            for k in range(m):
                result[i][j] += W[i][k] * X[k][j]
    return result


import torch

def to_tensors(W, X):
    W_t = torch.tensor(W)
    X_t = torch.tensor(X)
    return W_t, X_t


def vectorized_matmul(W_t, X_t):
    return torch.mm(W_t, X_t)


import timeit

plain_times = []
vec_times = []

for m in m_values:
    W, X = generate_matrices(m)
    W_t, X_t = to_tensors(W, X)

    t_plain = timeit.timeit(lambda: plain_matmul(W, X, m), number=3)
    t_vec   = timeit.timeit(lambda: vectorized_matmul(W_t, X_t), number=100)

    plain_times.append(t_plain / 3)
    vec_times.append(t_vec / 100)
    
    
import matplotlib.pyplot as plt

speedups = [p / v for p, v in zip(plain_times, vec_times)]

plt.plot(m_values, speedups, marker='o')
plt.xlabel('m (matrix inner dimension)')
plt.ylabel('Speedup Ratio (plain / vectorized)')
plt.title('Matrix Multiplication Speedup: Plain Python vs PyTorch Vectorized')
plt.grid(True)
plt.savefig('speedup_plot.png')
plt.show()


