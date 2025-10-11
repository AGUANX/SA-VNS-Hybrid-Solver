# DroneEnergyModel_Full.py
import os
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---------- 常量 ----------
BATTERY_J   = 539640
USABLE_FRAC = 0.80
MAX_FLIGHT_J = BATTERY_J * USABLE_FRAC
SPEED_MPS_Z   = 5
SPEED_MPS_XY   = 10
# 网格经度对应实际距离
PIXEL_M     = 30
H_RATE      = 9.36  # 水平
V_RATE      = 10.55  # 上升
ABS_V_RATE  = 9 # 下降
CHARGE_FULL_SEC = 0.5 * 3600
COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
          '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5']

# ---------- 权重 ----------
WEIGHT_TIME   = 0.3
WEIGHT_ENERGY = 0.01
WEIGHT_SORTIE = 1.0

# ---------- 归一化上限 ----------
MAX_TIME   = 86400 * 2      # 2 天
MAX_ENERGY = 2e8            # 200 MJ
MAX_SORTIE = 500            # 500 次

BASE_DIR = Path(__file__).parent

# ---------- 替换整个 evaluate_assignment_with_simulation ----------
def evaluate_assignment_with_simulation(file_path, assign_csv: Path,
                                      depot_csv: Path,
                                      output_dir: Path = None):
    # ---------- 原有加载 ----------
    # dem数据
    elev_arr   = np.loadtxt(BASE_DIR / 'nanling_final_matrix.csv', delimiter=',')
    assign_df  = pd.read_csv(assign_csv)
    # assign_df['depot_id'] = assign_df['depot_id'].replace(0, 6)
    depots_df  = pd.read_csv(depot_csv)
    depots_df['depot_id'] = depots_df.index + 1
    # 任务区域大小
    region_map = joblib.load(BASE_DIR / 'region_grid_map.pkl')

    # 场景文件
    risk_df = pd.read_csv(
        file_path,
        usecols=['region_id', 'rick_level'],  # 仅选这两列
        encoding='utf-8'
    ).astype({'region_id': int, 'rick_level': int})

    # 如果想把 'rick_level' 重命名为 'risk_flag'
    risk_df = risk_df.rename(columns={'rick_level': 'risk_flag'})
    high_risk_regions = set(risk_df[risk_df['risk_flag'] > 0]['region_id'])

    ROWS, COLS = elev_arr.shape
    def clamp(r, c): return max(0, min(r, ROWS - 1)), max(0, min(c, COLS - 1))
    def dist(r1, c1, r2, c2): return np.hypot(r1 - r2, c1 - c2) * PIXEL_M
    def energy(p1, p2):
        r1, c1 = clamp(*p1)
        r2, c2 = clamp(*p2)
        dz = elev_arr[r2, c2] - elev_arr[r1, c1]
        h_energy = 0
        if dz >= 0:
            h_energy = dz * V_RATE
        else:
            h_energy = dz * ABS_V_RATE
        dxy = dist(r1, c1, r2, c2)
        return dxy * H_RATE + h_energy, dxy / SPEED_MPS_XY + abs(dz) / SPEED_MPS_Z
    def get_pixels(region_id):
        return [region_map[k] for k in region_map.keys() if k[0] == region_id]
    # 生成任务区域路径
    def boustrophedon(mask):
        path = []
        for i in range(mask.shape[0]):
            idx = list(np.where(mask[i, :])[0])
            if not idx: continue
            path.extend([(i, j) for j in (idx if i % 2 == 0 else reversed(idx))])
        return path

    # ---------- 初始化 ----------
    depot_regions = {}
    for _, row in assign_df.iterrows():
        depot_id = int(row['depot_id'])
        region_id = int(row['region_id'])
        task_r = int(row['task_row'])
        task_c = int(row['task_col'])
        depot_regions.setdefault(depot_id, []).append((region_id, task_r, task_c))

    # 每个机巢时间列表
    total_time_sec_list = []
    # 能耗
    total_energy_j = 0
    # 轮次
    total_sorties  = 0
    now = datetime(2025, 7, 28, 8, 0, 0)

    # ---------- 日志 ----------
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        loggers = {d: [] for d in depot_regions.keys()}  # depot_id -> list[dict]

    def add_event(depot_id, event, r, c, energy_left, cum_energy, cum_time):
        if output_dir:
            loggers[depot_id].append({
                'timestamp': now.isoformat(timespec='seconds'),
                'event': event,
                'row': r,
                'col': c,
                'energy_left_J': round(energy_left),
                'cum_energy_J': round(cum_energy),
                'cum_time_sec': round(cum_time)
            })


    # ---------- 主循环 ----------
    for depot_id, region_list in depot_regions.items():
        # 统计单个机巢时间
        total_time_sec = 0
        nest_r = int(depots_df.loc[depot_id - 1, 'row'])
        nest_c = int(depots_df.loc[depot_id - 1, 'col'])
        region_list_sorted = sorted(
            region_list,
            key=lambda x: energy((nest_r, nest_c), (x[1], x[2]))
        )
        current_r, current_c = nest_r, nest_c
        total_energy = BATTERY_J
        sortie = 0

        for region_id, task_r, task_c in region_list_sorted:
            repeat = 2 if region_id in high_risk_regions else 1
            for loop in range(1, repeat + 1):
                pixels = get_pixels(region_id)
                if not pixels: continue
                rs, cs, _ = zip(*pixels)
                min_r, max_r = min(rs), max(rs)
                min_c, max_c = min(cs), max(cs)
                mask = np.zeros((max_r - min_r + 1, max_c - min_c + 1), bool)
                for r, c, _ in pixels:
                    mask[r - min_r, c - min_c] = True
                path_local = boustrophedon(mask)

                # 起飞（或再起飞）
                sortie += 1
                total_sorties += 1
                add_event(depot_id, 'takeoff', nest_r, nest_c, total_energy, total_energy_j, total_time_sec)

                # 前往任务区中心点
                e_entry, t_entry = energy((current_r, current_c), (task_r, task_c))
                now += timedelta(seconds=t_entry)
                total_time_sec += t_entry
                total_energy_j += e_entry
                total_energy -= e_entry
                add_event(depot_id, 'arrive_task', task_r, task_c, total_energy, total_energy_j, total_time_sec)
                current_r, current_c = task_r, task_c

                # 如需充电
                if total_energy < BATTERY_J * 0.2:
                    # 返航充电
                    e_back, t_back = energy((current_r, current_c), (nest_r, nest_c))
                    now += timedelta(seconds=t_back)
                    total_time_sec += t_back
                    total_energy_j += e_back
                    add_event(depot_id, 'return_charge', nest_r, nest_c, total_energy - e_back, total_energy_j, total_time_sec)

                    # 计算充电时间
                    charge_sec = max((BATTERY_J - (total_energy - e_back)) / (BATTERY_J * 0.8) * CHARGE_FULL_SEC, 0)
                    now += timedelta(seconds=charge_sec)
                    total_time_sec += charge_sec
                    add_event(depot_id, 'finish_charge', nest_r, nest_c, BATTERY_J, total_energy_j, total_time_sec)

                    # 返回返航点
                    e_go, t_go = energy((nest_r, nest_c), (current_r, current_c))
                    total_energy = BATTERY_J - e_go
                    total_time_sec += t_go
                    sortie += 1
                    total_sorties += 1
                    add_event(depot_id, 'takeoff', nest_r, nest_c, total_energy, total_energy_j, total_time_sec)

                # 拍照路径
                idx = 0
                while idx < len(path_local):
                    next_r = path_local[idx][0] + min_r
                    next_c = path_local[idx][1] + min_c
                    e_seg, t_seg = energy((current_r, current_c), (next_r, next_c))
                    remaining = total_energy - e_seg
                    if remaining < BATTERY_J * 0.2:
                        # 需返航充电
                        e_back, t_back = energy((current_r, current_c), (nest_r, nest_c))
                        now += timedelta(seconds=t_back)
                        total_time_sec += t_back
                        total_energy_j += e_back
                        add_event(depot_id, 'return_charge', nest_r, nest_c, total_energy - e_back, total_energy_j, total_time_sec)

                        charge_sec = max((BATTERY_J - (total_energy - e_back)) / (BATTERY_J * 0.8) * CHARGE_FULL_SEC, 0)
                        now += timedelta(seconds=charge_sec)
                        total_time_sec += charge_sec
                        add_event(depot_id, 'finish_charge', nest_r, nest_c, BATTERY_J, total_energy_j, total_time_sec)

                        # 返回返航点
                        e_go, t_go = energy((nest_r, nest_c), (current_r, current_c))
                        total_time_sec += t_go
                        total_energy = BATTERY_J - e_go
                        sortie += 1
                        total_sorties += 1
                        add_event(depot_id, 'takeoff', nest_r, nest_c, total_energy, total_energy_j, total_time_sec)
                        continue

                    total_energy -= e_seg
                    current_r, current_c = next_r, next_c
                    idx += 1
                    total_time_sec += t_seg
                    total_energy_j += e_seg
                    add_event(depot_id, 'photo_point', current_r, current_c, total_energy, total_energy_j, total_time_sec)

                # 返航
                e_back, t_back = energy((current_r, current_c), (nest_r, nest_c))
                now += timedelta(seconds=t_back)
                total_time_sec += t_back
                total_energy_j += e_back
                total_energy -= e_back
                add_event(depot_id, 'land', nest_r, nest_c, total_energy, total_energy_j, total_time_sec)

        total_time_sec_list.append(total_time_sec)

    # ---------- 归一化 ----------
    max_time_sec = max(total_time_sec_list)
    min_time_sec = min(total_time_sec_list)
    # norm_energy = min(total_energy_j / MAX_ENERGY, 1.0)
    # norm_sortie = min(total_sorties / MAX_SORTIE, 1.0)d
    norm_energy = total_energy_j * 0.5 / 3600000
    # norm_sortie = total_sorties * 2.5
    time_rate = float(max_time_sec / min_time_sec)
    score_norm = (norm_energy + total_sorties / 100) * time_rate

    # ---------- 输出 ----------
    if output_dir:
        for depot_id, rows in loggers.items():
            pd.DataFrame(rows).to_csv(output_dir / f"drone_log_depot{depot_id}.csv", index=False)
        # 再写全局摘要
        with open(output_dir / "mission_summary.txt", "w", encoding="utf-8") as f:
            f.write("=== Mission Summary ===\n")
            f.write(f"Total time (s): {total_time_sec}\n")
            f.write(f"Total energy (J): {total_energy_j}\n")
            f.write(f"Total sorties: {total_sorties}\n")
            f.write(f"Normalized score: {score_norm}\n")

    return time_rate, total_energy_j, total_sorties, score_norm, max_time_sec