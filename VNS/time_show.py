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

K = 6

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



# ---------- 分配 ----------
def assign_nearest(tasks, depots):
    assign = np.empty(len(tasks), dtype=int)
    for i, (r, c) in enumerate(tasks[['row', 'col']].values):
        dists = [np.hypot(r - d[0], c - d[1]) for d in depots]
        assign[i] = np.argmin(dists)
    return assign



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



# ---------- 主 ----------
def main():
    tasks  = load_tasks()

    # 初始机巢（落在高程有效区）
    depots = [(147, 322), (495, 316), (432, 80), (310, 297), (421, 547), (183, 173)]
    print("初始机巢解", depots)
    best_t, best_e, best_s, best_score, max_time= compute_depot(depots, tasks)


    log(f"✅ 完成，最优得分={best_score:.4f}  飞行时间负载={best_t}s  能耗={best_e:,.0f}J  架次={best_s}  最大时间={max_time}", "green")

if __name__ == '__main__':
    main()

