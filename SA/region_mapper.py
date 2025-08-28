import pandas as pd
import os

# 构建映射表（只运行一次）
root = r".."
task_dir = os.path.join(root, "task_regions_final")
map_path = os.path.join(root, "region_grid_map.pkl")

def build_map():
    """一次性生成映射表并保存为 pkl，提高后续调用速度"""
    import joblib
    import numpy as np

    mapping = {}
    for region_id in range(1, 187):
        grid_file = f"region_{region_id}_grid.csv"
        pixel_file = f"region_{region_id}_pixels.csv"
        if not (os.path.exists(os.path.join(task_dir, grid_file)) and
                os.path.exists(os.path.join(task_dir, pixel_file))):
            continue

        # 计算左上角偏移（大网格行列号）
        pixels = pd.read_csv(os.path.join(task_dir, pixel_file))
        offset_row = pixels['row'].min()
        offset_col = pixels['col'].min()

        # 构建字典： (region_id, row_small, col_small) -> (row_big, col_big, elevation)
        grid = np.loadtxt(os.path.join(task_dir, grid_file), delimiter=',')
        rows, cols = grid.shape
        for i in range(rows):
            for j in range(cols):
                if grid[i, j] != -999:
                    key = (region_id, i, j)
                    val = (offset_row + i, offset_col + j, grid[i, j])
                    mapping[key] = val

    joblib.dump(mapping, map_path)
    print("映射表已生成，下次直接加载即可。")

# 第一次运行时生成映射表（只需一次）
if not os.path.exists(map_path):
    build_map()

# ===== 可调用的函数 =====
import joblib
_mapping = joblib.load(map_path)

def region_to_big(region_id: int, row_small: int, col_small: int):
    """
    输入：任务区域 ID + 区域内部行列号
    返回：(row_big, col_big, elevation)
    若不存在返回 None
    """
    return _mapping.get((region_id, row_small, col_small), None)

# ===== 示例 =====
if __name__ == "__main__":
    print(region_to_big(10, 0, 0))  # 区域 1 左上角 → (大行列, 高程)