import joblib
import numpy as np
import pandas as pd
from fontTools.misc.cython import returns
from osgeo import gdal
import math
import time


BATTERY_J   = 539640
SPEED_MPS_Z   = 5
SPEED_MPS_XY   = 10
# 网格经度对应实际距离
PIXEL_M     = 30
H_RATE      = 9.36  # 水平
V_RATE      = 10.55  # 上升
ABS_V_RATE  = 9 # 下降


class UAV:
    # 能耗系数 水平能耗k_s 垂直能耗 k_c
    k_s = 9.36
    k_c = 10.55


def create_data(r):
    dx = np.arange(-r, r + 1)
    dy = np.arange(-r, r + 1)
    DX, DY = np.meshgrid(dx, dy)
    points = np.array([DX.flatten(), DY.flatten()]).T
    return points


def rotate_3d_map(points, angle):
    angle_rad = math.radians(angle)
    R = np.array([
        [math.cos(angle_rad), math.sin(angle_rad)],
        [-math.sin(angle_rad), math.cos(angle_rad)]
    ])
    rotated_points = np.dot(points, R)
    size = int(math.sqrt(len(points)))  # 2r+1
    return rotated_points[:, 0].reshape(size, size), rotated_points[:, 1].reshape(size, size)


def check_points_in_range(X_rotated, Y_rotated, nrows, ncols):
    mask = (X_rotated >= 0) & (X_rotated < ncols) & (Y_rotated >= 0) & (Y_rotated < nrows)
    return mask


def hight_interpolation(X_rotated, Y_rotated, nrows, ncols, Z):
    mask = check_points_in_range(X_rotated, Y_rotated, nrows, ncols)
    hight = np.zeros_like(X_rotated, dtype=float)

    for i in range(X_rotated.shape[0]):
        for j in range(X_rotated.shape[1]):
            if mask[i][j]:
                x = X_rotated[i][j]
                y = Y_rotated[i][j]
                x_floor = int(np.floor(x))
                x_ceil = min(x_floor + 1, ncols - 1)
                y_floor = int(np.floor(y))
                y_ceil = min(y_floor + 1, nrows - 1)

                dx = x - x_floor
                dy = y - y_floor

                h1 = Z[y_floor, x_floor]
                h2 = Z[y_floor, x_ceil]
                h3 = Z[y_ceil, x_floor]
                h4 = Z[y_ceil, x_ceil]

                if any(np.isnan([h1, h2, h3, h4])):
                    hight[i, j] = np.nan
                else:
                    hight[i, j] = (h1 * (1 - dx) * (1 - dy) +
                                   h2 * dx * (1 - dy) +
                                   h3 * (1 - dx) * dy +
                                   h4 * dx * dy)
            else:
                hight[i][j] = np.nan
    mask = ~np.isnan(hight)
    return mask, hight


def boustrophedon_path(mask):
    path = []
    for i in range(mask.shape[0]):
        if i % 2 == 0:
            for j in range(mask.shape[1]):
                if mask[i][j]:
                    path.append((i, j))
        else:
            for j in range(mask.shape[1] - 1, -1, -1):
                if mask[i][j]:
                    path.append((i, j))
    return path


def energy(dx, dy, dz):
    h_energy = 0
    if dz >= 0:
        h_energy = dz * V_RATE
    else:
        h_energy = dz * ABS_V_RATE
    dxy = math.hypot(dx, dy) * PIXEL_M
    return dxy * H_RATE + h_energy


def calculate_path(path, hight, t_energy):
    if not path:
        return 0
    total = 0.0
    back = []
    for k in range(1, len(path)):
        i1, j1 = path[k - 1]
        i2, j2 = path[k]
        z1 = hight[i1][j1]
        z2 = hight[i2][j2]
        dx = i2 - i1
        dy = j2 - j1
        dz = z2 - z1
        return_total = energy(dx, dy, dz)
        total += return_total
        if total >= (t_energy - BATTERY_J * 0.2):
            back.append(path[k])
        total = return_total

    return total, path

def get_pixels(region_id):
    region_map = joblib.load('region_grid_map.pkl')
    return [region_map[k] for k in region_map.keys() if k[0] == region_id]


def rotated_calculate(i, energy):
    m = get_pixels(i)
    rs, cs, l = zip(*m)
    min_r, max_r = min(rs), max(rs)
    min_c, max_c = min(cs), max(cs)
    Z = np.full((max_r - min_r + 1, max_c - min_c + 1), np.nan, dtype=object)
    for r, c, l in m:
        Z[r - min_r, c - min_c] = l
    nrows, ncols = Z.shape
    cx = ncols / 2.0
    cy = nrows / 2.0
    r = math.ceil(math.sqrt((ncols / 2) ** 2 + (nrows / 2) ** 2))
    points = create_data(r)

    row_i = pd.read_csv('rotated_data.csv', skiprows=1, nrows=1, header=None).iloc[0]
    angle = row_i[1]
    dx_rot, dy_rot = rotate_3d_map(points, angle)
    X_rot = dx_rot + cx
    Y_rot = dy_rot + cy
    mask, hight = hight_interpolation(X_rot, Y_rot, nrows, ncols, Z)
    path = boustrophedon_path(mask)
    length, back = calculate_path(path, hight, energy)

    return back

if __name__ == '__main__':
    print(rotated_calculate(1))