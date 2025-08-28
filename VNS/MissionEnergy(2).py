# MissionEnergy.py  2025-06-08 终极版
# 1. 仅允许机巢落在有效高程区域（跳过 -999）
# 2. 逐事件日志 + 3D 可视化
import math
import os
import random
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import seaborn as sns
from datetime import datetime
from DroneEnergyModel_Full import evaluate_assignment_with_simulation

# ---------- 路径 ----------
BASE_DIR   = Path(__file__).parent
MAP_CSV    = BASE_DIR / 'nanling_final_matrix.csv'
# 任务区域中心点
CENTER_CSV = BASE_DIR / 'task_region_centers.csv'

RISK_CSV   = BASE_DIR / '场景1.csv'

ENERGY_CSV = BASE_DIR / 'result.csv'
ASSIGN_CSV = BASE_DIR / 'assignment_fix.csv'
DEPOT_CSV  = BASE_DIR / 'depots_fix.csv'
BEST_DIR   = BASE_DIR / '最优一代结果'
BEST_DIR.mkdir(exist_ok=True)

K   = 6
GENS = 50

# ---------- 彩色日志 ----------
def log(msg, color="white"):
    colors = {"red": 91, "green": 92, "yellow": 93, "blue": 94, "magenta": 95}
    print(f"\033[{colors.get(color, 97)}m{msg}\033[0m")

# ---------- 数据 ----------
def load_tasks():

    center = pd.read_csv(CENTER_CSV, encoding='utf-8-sig').rename(columns={'task_row': 'row', 'task_col': 'col'})
    # 场景文件
    risk = pd.read_csv(RISK_CSV, header=0, names=['risk_flag', 'region_id'], encoding='gbk') \
             .fillna(0).astype(int)
    energy = pd.read_csv(ENERGY_CSV, encoding='utf-8')[['id', '区域能耗']]
    task = center.merge(risk, on='region_id') \
                 .merge(energy, left_on='region_id', right_on='id') \
                 .drop_duplicates('region_id')
    task['risk_weight'] = np.where(task['risk_flag'] > 0, 2, 1)
    return task[['region_id', 'row', 'col', '区域能耗', 'risk_weight']]

# ---------- 有效边界 ----------
def get_bounds():
    elev = np.loadtxt(MAP_CSV, delimiter=',')
    valid = np.where(elev != -999)
    r_min, r_max = int(valid[0].min()), int(valid[0].max())
    c_min, c_max = int(valid[1].min()), int(valid[1].max())
    return r_min, r_max, c_min, c_max, elev

# ---------- 合法随机坐标 ----------
def random_valid_coord(r_min, r_max, c_min, c_max, elev):
    while True:
        r = random.randint(r_min, r_max)
        c = random.randint(c_min, c_max)
        if elev[r, c] != -999:
            return r, c

# ---------- 分配 ----------
def assign_nearest(tasks, depots):
    assign = np.empty(len(tasks), dtype=int)
    for i, (r, c) in enumerate(tasks[['row', 'col']].values):
        dists = [np.hypot(r - d[0], c - d[1]) for d in depots]
        assign[i] = np.argmin(dists)
    return assign

# ---------- 3D 可视化 ----------
def plot_3d_solution(depots, tasks, assign):
    depots = np.array(depots)
    tasks_xy = tasks[['row', 'col']].values
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
        mask = assign == depot_id
        ax.scatter(tasks_xy[mask, 1], tasks_xy[mask, 0],
                   elev[tasks_xy[mask, 0], tasks_xy[mask, 1]] + 100,
                   color=palette[depot_id], s=35, label=f'D{depot_id+1}')
    ax.set_xlabel('Col'); ax.set_ylabel('Row'); ax.set_zlabel('Elevation (m)')
    ax.set_title('VNS Optimal 3D View'); ax.legend()
    plt.tight_layout()
    plt.savefig(BEST_DIR / 'optimal_3d.png', dpi=300)
    plt.show()


# 机巢扰动
def disturbance(points):
    points = np.array([list(point) for point in points])
    new_dep = []
    r_min, r_max, c_min, c_max, elev = get_bounds()
    for i in range(len(points)):
        radius = 50
        direction = [random.randint(-radius, radius), random.randint(-radius, radius)]

        new_r = max(r_min, min(r_max, int(points[i][0] +  direction[0])))
        new_c = max(c_min, min(c_max, int(points[i][1] +  direction[1])))
        # 若越界或-999，则小范围位移
        while not (r_min <= new_r <= r_max and c_min <= new_c <= c_max) or elev[new_r, new_c] == -999:
            if new_r < r_min:
                new_r = r_max
            if new_c < c_min:
                 new_c = c_max
            if new_r > r_max:
                new_r = r_min
            if new_c > c_max:
                new_c = c_min
            if elev[new_r, new_c] == -999:
                new_r += 1
                new_c += 1
        new_dep.append((new_r, new_c))
    return new_dep

# 输入机巢布局，输出t, e, s, score_norm、
def compute_depot(cand, tasks):
    tmp = BEST_DIR / '_tmp';tmp.mkdir(exist_ok=True)
    pd.DataFrame(cand, columns=['row', 'col']).assign(depot_id=range(1, K + 1)) \
        .to_csv(tmp / 'depots_fix.csv', index=False)
    assign = assign_nearest(tasks, cand)
    pd.DataFrame({
        'region_id': tasks['region_id'],
        'depot_id': assign + 1,
        'task_row': tasks['row'],
        'task_col': tasks['col']
    }).to_csv(tmp / 'assignment_fix.csv', index=False, encoding='utf-8-sig')

    t, e, s, score_norm, max_time = evaluate_assignment_with_simulation(
        tmp / 'assignment_fix.csv',
        tmp / 'depots_fix.csv'
    )
    return t, e, s, score_norm, max_time


# 成本收敛曲线
def plot_score(scores):
    # 绘制成本收敛图
    plt.plot(scores)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Simulated Annealing Cost Convergence")
    plt.savefig("cost.png")
    plt.show()



# ---------- 主 ----------
def main():
    tasks  = load_tasks()
    r_min, r_max, c_min, c_max, elev = get_bounds()

    # 初始机巢（落在高程有效区）
    depots = [random_valid_coord(r_min, r_max, c_min, c_max, elev) for _ in range(K)]
    print("初始机巢解", depots)
    best_t, best_e, best_s, best_score, max_time = compute_depot(depots, tasks); best_dep = depots.copy(); best_dep_result = depots.copy()

    # 模拟退火算法的收敛
    scores = []

    # vns算法的迭代次数
    max_iterations = 200

    gen = 0
    pbar = tqdm(total=max_iterations, desc='VNS')  # 总长度,标题，说明
    while gen  < max_iterations:
        pbar.update(1)  # 步进长度
        gen += 1
        # 机巢扰动
        new_dep = disturbance(best_dep)
        log(f"candidates:{new_dep}")

        t, e, s, score_norm, time = compute_depot(new_dep, tasks)

        if score_norm < best_score:
            best_t, best_e, best_s, best_score, best_dep, max_time = t, e, s, score_norm, new_dep, time
            log(f"[Gen {gen:02d}] 🎯 更新最优解！得分={best_score:.4f}", "green")
            # 重启迭代
            gen = 1
            pbar.reset()

        log(f"[Gen {gen:02d}] 目标={best_score:.4f}  "
            f"飞行时间负载={best_t}s  能耗={best_e:,.0f}J  架次={best_s}   最大时间={max_time}s", "yellow")
        scores.append(best_score)


    # 最终文件 & 日志 & 3D
    # 使用best_dep_result保存真正的最佳结果
    assign = assign_nearest(tasks, best_dep)
    pd.DataFrame(best_dep, columns=['row', 'col']).assign(depot_id=range(1, K + 1)) \
        .to_csv(DEPOT_CSV, index=False)
    pd.DataFrame({
        'region_id': tasks['region_id'],
        'depot_id': assign + 1,
        'task_row': tasks['row'],
        'task_col': tasks['col']
    }).to_csv(ASSIGN_CSV, index=False, encoding='utf-8-sig')

    evaluate_assignment_with_simulation(
        ASSIGN_CSV,
        DEPOT_CSV,
        output_dir=BEST_DIR
    )
    plot_3d_solution(best_dep, tasks, assign)

    plot_score(scores)
    log(f"✅ 完成，最优得分={best_score:.4f}  飞行时间负载={best_t}s  能耗={best_e:,.0f}J  架次={best_s}", "green")

if __name__ == '__main__':
    main()




