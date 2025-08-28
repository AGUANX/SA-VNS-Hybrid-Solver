import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


from pathlib import Path

# ---------- 路径 ----------
BASE_DIR   = Path(__file__).parent
MAP_CSV    = BASE_DIR / 'nanling_final_matrix.csv'
BEST_DIR   = BASE_DIR / '最优一代结果'
ASSIGN_CSV = BASE_DIR / 'assignment_fix.csv'
DEPOT_CSV  = BASE_DIR / 'depots_fix.csv'


# ---------- 3D 可视化 ----------
def plot_3d_solution(depots, tasks, assign):
    depots = np.array(depots)
    tasks_xy = tasks[['task_row', 'task_col']].values
    elev = np.loadtxt(MAP_CSV, delimiter=',')
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(elev.shape[1])
    y = np.arange(elev.shape[0])
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, elev, cmap='terrain', alpha=0.4, rstride=10, cstride=10)
    ax.scatter(depots[:, 1], depots[:, 0],
               elev[depots[:, 0], depots[:, 1]] + 100,
               c='red', s=250, marker='^', edgecolors='k', label='Depot')
    palette = sns.color_palette("tab10", len(depots))
    for depot_id in range(len(depots)):
        mask = [assign[i] == int(depot_id) for i in range(len(assign))]
        print(mask)
        ax.scatter(tasks_xy[mask, 1], tasks_xy[mask, 0],
                   elev[tasks_xy[mask, 0], tasks_xy[mask, 1]] + 100,
                   color=palette[depot_id], s=35, label=f'D{depot_id+1}')
    ax.set_xlabel('Col'); ax.set_ylabel('Row'); ax.set_zlabel('Elevation (m)')
    ax.set_title('VNS Optimal 3D View'); ax.legend()
    plt.tight_layout()
    plt.savefig(BEST_DIR / 'optimal_3d.png', dpi=300)
    plt.show()



# 读取 DEPOT_CSV 文件来恢复 best_dep_result
depot_df = pd.read_csv(DEPOT_CSV)
best_dep_result = depot_df[["row", "col"]].values.tolist()

# 读取 ASSIGN_CSV 文件来恢复 tasks 和 assign
assign_df = pd.read_csv(ASSIGN_CSV)
tasks = assign_df[["region_id", "task_row", "task_col"]]
assign = (assign_df["depot_id"] - 1).values.tolist()

plot_3d_solution(best_dep_result, tasks, assign)
print("best_dep_result:", best_dep_result)
print("tasks:", tasks)
print("assign:", assign)
print("数据已成功恢复并验证")


