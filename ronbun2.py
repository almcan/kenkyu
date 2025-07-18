import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def sir_model(t, y, M, N, alpha, beta, mu):
    """
    Args:
        t (float): 時間
        y (array): 状態変数 [S, I, R]
        M (int): グリッドの行数
        N (int): グリッドの列数
        alpha (float): 感染率
        beta (float): 除去率    
        mu (float): 感染者の移動率
    Returns:
        np.array: 状態変数の変化 [dS/dt, dI/dt, dR/dt]
    """
    #１次配列yをS,I,Rの二次元グリッドに変換
    S = y[:M*N].reshape((M, N))
    I = y[M*N:2*M*N].reshape((M, N))
    R = y[2*M*N:].reshape((M, N))

    #1.隣接セルの感染者総数
    kernel = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]])
    infected_neighbors_sum = convolve2d(I, kernel, mode='same', boundary='fill', fillvalue=0)
    
    #2.隣接セルAからの移動
    adjacent_counts = np.full((M, N), 8.0)
    adjacent_counts[0, :] -= 3
    adjacent_counts[-1, :] -= 3
    adjacent_counts[:, 0] -= 3
    adjacent_counts[:, -1] -= 3
    adjacent_counts[0, 0] += 2
    adjacent_counts[0, -1] += 2
    adjacent_counts[-1, 0] += 2
    adjacent_counts[-1, -1] += 2

    S_norm = S / adjacent_counts
    I_norm = I / adjacent_counts
    R_norm = R / adjacent_counts

    mobility_S = convolve2d(S_norm, kernel, mode='same', boundary='fill', fillvalue=0) - S_norm
    mobility_I = convolve2d(I_norm, kernel, mode='same', boundary='fill', fillvalue=0) - I_norm
    mobility_R = convolve2d(R_norm, kernel, mode='same', boundary='fill', fillvalue=0) - R_norm

    #dS/dt, dI/dt, dR/dtの計算
    dS_dt = -alpha * S * infected_neighbors_sum - mu * S + mu * mobility_S
    dI_dt = alpha * S * infected_neighbors_sum - beta * I - mu * I + mu * mobility_I
    dR_dt = beta * I - mu * R + mu * mobility_R

    return np.concatenate([dS_dt.ravel(), dI_dt.ravel(), dR_dt.ravel()])

M, N = 100, 100 # グリッドのサイズ
K = 5000        # ホスト数
alpha = 0.05    # 感染率
beta = 0.01     # 回復率
mu = 1.0        # 移動率

S0 = np.zeros((M, N))
I0 = np.zeros((M, N))
R0 = np.zeros((M, N))

susceptible_indices = np.random.choice(M * N, K - 1, replace=False)
rows ,cols = np.unravel_index(susceptible_indices, (M, N))
S0[rows, cols] = 1

initial_infected_pos = (50, 50) # 初期感染者の位置
S0[initial_infected_pos] = 0
I0[initial_infected_pos] = 1

y0 = np.concatenate([S0.ravel(), I0.ravel(), R0.ravel()])
t_span = (0, 800)  # シミュレーション時間
t_eval = np.linspace(t_span[0], t_span[1], 200)  # 評価時間

print("Starting SIR model simulation...")
solution = solve_ivp(
    sir_model, 
    t_span, 
    y0, 
    args=(M, N, alpha, beta, mu), 
    t_eval=t_eval, 
    method='RK45'
)
print("SIR model simulation completed.")
# 結果の可視化
S_total = solution.y[:M*N].sum(axis=0)
I_total = solution.y[M*N:2*M*N].sum(axis=0)
R_total = solution.y[2*M*N:].sum(axis=0)

plt.figure(figsize=(10, 6))
plt.plot(solution.t, S_total, label='S(Susceptible)')
plt.plot(solution.t, I_total, label='I(Infected)')
plt.plot(solution.t, R_total, label='R(Recovered)')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model Simulation Results')
plt.legend()
plt.grid(True)
plt.savefig('ronbun2_plot.png')
