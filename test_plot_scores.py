import numpy as np
from matplotlib import pyplot as plt


# 成本收敛曲线
def plot_score(scores):
    # 绘制成本收敛图
    plt.plot(scores)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Simulated Annealing Cost Convergence")
    plt.show()


# 生成测试数据
np.random.seed(42)  # 设置随机种子以保证结果可复现
iterations = 50  # 迭代次数
initial_score = 100.0  # 初始分数
scores = initial_score - np.arange(iterations) + 10 * np.sin(np.linspace(0, 6 * np.pi, iterations)) + np.random.normal(0, 3, iterations)

plot_score(scores)

