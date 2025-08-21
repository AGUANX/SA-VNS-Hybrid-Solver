import pandas as pd

# 读取 CSV 文件
data = pd.read_csv('nanling_final_matrix.csv', header=None)

# 初始化边界点列表
boundary_points = []
rows, cols = data.shape

# 遍历每个点，检查是否为边界点
for i in range(rows):
    for j in range(cols):
        is_boundary = False
        if data.iloc[i, j] == -999:
            continue
        # 左邻居
        if j > 0 and data.iloc[i, j - 1] == -999:
            is_boundary = True
        # 右邻居
        if j < cols - 1 and data.iloc[i, j + 1] == -999:
            is_boundary = True
        # 上邻居
        if i > 0 and data.iloc[i - 1, j] == -999:
            is_boundary = True
        # 下邻居
        if i < rows - 1 and data.iloc[i + 1, j] == -999:
            is_boundary = True
        if is_boundary:
            boundary_points.append((i, j))

# 将边界点保存为新的 CSV 文件
boundary_df = pd.DataFrame(boundary_points, columns=['Row', 'Column'])
boundary_df.to_csv('boundary_points.csv', index=False)

print("边界点已保存到 boundary_points.csv 文件中！")