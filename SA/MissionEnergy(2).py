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

# 模拟退火算法接收新解
def accept(delta, temperature):
    return math.exp(-delta / temperature) > random.random()


# 机巢扰动，使用弹簧模型加上边界约束：
def disturbance(points, T, boundarys = pd.read_csv("boundary_points.csv").values):
    points = np.array([list(point) for point in points])
    new_dep = []
    r_min, r_max, c_min, c_max, elev = get_bounds()
    for i in range(len(points)):
        direction = (0, 0)
        min_distance = float('inf')
        for j in range(len(points)):
            if i != j:
                # 计算点与点之间的力
                # direction += points[j] - points[i]
                distance = np.linalg.norm(points[i] - points[j])
                min_distance = min(distance, min_distance)
                if min_distance == distance:
                    min_direction = points[i] - points[j]
        direction = min_direction * math.exp(-min_distance)
        # 计算边界的推力，先找到最近的边界点，然后计算距离，如果小于机巢点之间的最小距离的二分之一，则表示机巢点距离边界太近了，计算方向，把距离乘二加入合力。不约束机巢距离边界太远
        min_distance_boundary = float('inf')
        for index, boundary in enumerate(boundarys):
            dis = np.linalg.norm(boundary - points[i])
            if dis < min_distance_boundary:
                min_distance_boundary = dis
                min_index = index
        if min_distance/2 < min_distance_boundary:
            direction += (boundarys[min_index] - points[i]) * math.exp(-min_distance/2) * 2
        else:
            direction += (points[i] - boundarys[min_index]) * math.exp(-min_distance_boundary) * 2
        norm = np.linalg.norm(direction)
        if norm == 0:
            direction_norm = 0
        else:
            direction_norm = direction / norm


        radius = int(T)
        dr = random.randint(0, radius)

        new_r = max(r_min, min(r_max, int(points[i][0] + dr * direction_norm[0])))
        new_c = max(c_min, min(c_max, int(points[i][1] + dr * direction_norm[1])))
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
    # depots = [(round((r_min + r_max)/2), round((c_min + c_max)/2)) for _ in range(K)]
    # depots = [(r_min, c_min) for _ in range(K)]
    print("初始机巢解", depots)
    best_score = float('inf'); best_dep = depots.copy(); best_dep_result = depots.copy()
    # 模拟退火算法 温度 衰退率
    T = 100
    t_l = 0.98
    target = 1
    total_iterations = 0
    temp = T
    while temp > target:
        temp *= t_l
        total_iterations += 1
    GENS = total_iterations

    # 模拟退火算法的收敛
    scores = []


    pbar = tqdm(range(1, GENS + 1), desc='VNS')
    for gen in pbar:
        candidates = []
        for _ in range(5):
            # 机巢扰动
            new_dep = disturbance(best_dep, T)
            candidates.append(new_dep)
        log(f"candidates:{candidates}")

        best_gen = None; best_gen_score = float('inf')
        for cand in candidates:
            tmp = BEST_DIR / '_tmp'; tmp.mkdir(exist_ok=True)
            pd.DataFrame(cand, columns=['row', 'col']).assign(depot_id=range(1, K + 1)) \
                .to_csv(tmp / 'depots_fix.csv', index=False)
            assign = assign_nearest(tasks, cand)
            pd.DataFrame({
                'region_id': tasks['region_id'],
                'depot_id': assign + 1,
                'task_row': tasks['row'],
                'task_col': tasks['col']
            }).to_csv(tmp / 'assignment_fix.csv', index=False, encoding='utf-8-sig')

            t, e, s, score_norm = evaluate_assignment_with_simulation(
                tmp / 'assignment_fix.csv',
                tmp / 'depots_fix.csv'
            )
            if score_norm < best_gen_score:
                best_gen_score, best_gen = score_norm, cand
                best_t, best_e, best_s = t, e, s
            # else:
            #     if accept(score_norm - best_gen_score, T):
            #         best_gen_score, best_gen = score_norm, cand
            #         log(f"[Gen {gen:02d}] 🎯 接受劣解！得分={best_gen_score:.4f}", "green")
            # shutil.rmtree(tmp, ignore_errors=True)
            log(f"[Gen {gen:02d}] 飞行时间负载={t}s  能耗={e:,.0f}J  架次={s} 得分={score_norm:.4f}", "green")

        if best_gen_score < best_score:
            best_score, best_dep = best_gen_score, best_gen
            best_dep_result = best_dep
            log(f"[Gen {gen:02d}] 🎯 更新最优解！得分={best_score:.4f}", "green")
        else:
            if accept(best_gen_score - best_score, T):
                best_score, best_dep = best_gen_score, best_gen
                log(f"[Gen {gen:02d}] 🎯 接受劣解！得分={best_gen_score:.4f}", "green")
        log(f"[Gen {gen:02d}] 目标={best_score:.4f}  "
            f"飞行时间负载={best_t}s  能耗={best_e:,.0f}J  架次={best_s}  当前温度={T}", "yellow")
        scores.append(best_score)
        T = T*t_l

    # 最终文件 & 日志 & 3D
    # 使用best_dep_result保存真正的最佳结果
    assign = assign_nearest(tasks, best_dep_result)
    pd.DataFrame(best_dep_result, columns=['row', 'col']).assign(depot_id=range(1, K + 1)) \
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
    plot_3d_solution(best_dep_result, tasks, assign)

    plot_score(scores)
    t, e, s, score_norm = evaluate_assignment_with_simulation('assignment_fix.csv', 'depots_fix.csv')
    log(f"✅ 完成，最优得分={score_norm:.4f}  飞行时间负载={t}s  能耗={e:,.0f}J  架次={s}", "green")

if __name__ == '__main__':
    main()





    # ---------- 论文扩展 ----------
    def run_batch(k_list):
        """批量跑不同 K，生成论文级结果"""
        tasks = load_tasks()
        r_min, r_max, c_min, c_max, elev = get_bounds()
        result_root = BASE_DIR / 'paper_results'
        result_root.mkdir(exist_ok=True)

        for k in k_list:
            run_one(k, tasks, r_min, r_max, c_min, c_max, elev, result_root)


    def run_one(k, tasks, r_min, r_max, c_min, c_max, elev, root):
        dir_k = root / f'K={k}';
        dir_k.mkdir(exist_ok=True)
        depots = [random_valid_coord(r_min, r_max, c_min, c_max, elev) for _ in range(k)]
        best_score = float('inf');
        best_dep = depots.copy()
        curve = []

        for gen in tqdm(range(1, GENS + 1), desc=f'K={k}'):
            new_dep = [random_valid_coord(r_min, r_max, c_min, c_max, elev) for _ in range(k)]
            pd.DataFrame(new_dep, columns=['row', 'col']).assign(depot_id=range(1, k + 1)) \
                .to_csv(dir_k / 'depots_fix.csv', index=False)
            assign = assign_nearest(tasks, new_dep)
            pd.DataFrame({
                'region_id': tasks['region_id'],
                'depot_id': assign + 1,
                'task_row': tasks['row'],
                'task_col': tasks['col']
            }).to_csv(dir_k / 'assignment_fix.csv', index=False, encoding='utf-8-sig')

            t, e, s, score_norm = evaluate_assignment_with_simulation(
                dir_k / 'assignment_fix.csv',
                dir_k / 'depots_fix.csv',
                output_dir=dir_k
            )
            if score_norm < best_score:
                best_score, best_dep = score_norm, new_dep
            curve.append({'gen': gen, 'score': score_norm,
                          'time': t, 'energy': e, 'sorties': s})

        # 保存核心数据
        pd.DataFrame(curve).to_csv(dir_k / 'cost_curve.csv', index=False)

        assign = assign_nearest(tasks, best_dep)
        summary = pd.DataFrame({
            'depot_id': range(1, k + 1),
            'row': [d[0] for d in best_dep],
            'col': [d[1] for d in best_dep],
            'task_count': np.bincount(assign, minlength=k),
            'total_energy': [e for e in np.zeros(k)],  # 占位，后续用日志汇总
            'total_sorties': [s for s in np.zeros(k)]
        })
        summary.to_csv(dir_k / 'summary.csv', index=False)

        # 绘图
        plot_paper_figs(dir_k, curve, best_dep, assign)


    def plot_paper_figs(dir_k, curve, best_dep, assign):
        # 1. 成本曲线
        plt.figure(figsize=(10, 4))
        plt.plot([c['gen'] for c in curve], [c['score'] for c in curve])
        plt.title(f'成本迭代曲线 K={dir_k.name}')
        plt.xlabel('迭代');
        plt.ylabel('归一化得分')
        plt.savefig(dir_k / 'cost_curve.png', dpi=300);
        plt.close()

        # 2. 工作量柱状图
        df = pd.read_csv(dir_k / 'summary.csv')
        plt.figure(figsize=(6, 4))
        plt.bar(df['depot_id'], df['task_count'], color='steelblue')
        plt.title(f'机巢任务数对比 K={dir_k.name}')
        plt.xlabel('机巢');
        plt.ylabel('任务数');
        plt.savefig(dir_k / 'workload_bar.png', dpi=300);
        plt.close()

        # 3. 高程布局图
        plt.figure(figsize=(7, 7))
        elev_plot = np.where(ELEV == -999, np.nan, ELEV)
        plt.imshow(elev_plot, cmap='terrain', origin='lower')
        plt.colorbar(label='Elevation (m)')
        plt.scatter([c for r, c in best_dep], [r for r, c in best_dep],
                    c='red', marker='X', s=120)
        plt.title(f'机巢布局 K={dir_k.name}');
        plt.savefig(dir_k / 'depot_layout.png', dpi=300);
        plt.close()


