"""
twi_irregular.py
计算不规则边界 DEM 的 Topographic Wetness Index（TWI）
无效像元用 -999 标记
"""
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


# ------------------------------------------------------------------
# 1. 生成虚假 DEM（仅供测试，用你自己的数据时把 dem 换掉即可）
# ------------------------------------------------------------------
def fake_dem(shape=(120, 120), dx=30.0, seed=42):
    rng = np.random.default_rng(seed)
    ny, nx = shape
    x, y = np.meshgrid(np.linspace(0, nx-1, nx), np.linspace(0, ny-1, ny))

    main_slope = 0.3 * (x + y)                          # 主坡向
    valley = 20 * np.exp(-((x - y)**2) / (2 * 15**2))   # V 形谷
    hill1 = 40 * np.exp(-((x - 0.3*nx)**2 + (y - 0.7*ny)**2) / (2 * 12**2))
    hill2 = 35 * np.exp(-((x - 0.7*nx)**2 + (y - 0.3*ny)**2) / (2 * 10**2))
    noise = 2.0 * rng.normal(size=shape)

    dem = main_slope - valley + hill1 + hill2 + noise
    dem -= dem.min()
    return dem


# ------------------------------------------------------------------
# 2. 计算 TWI（支持 nodata 屏蔽）
# ------------------------------------------------------------------
def calculate_twi(dem, cell_size=30.0, tolerance=1e-6, nodata=-999):
    """返回 TWI 数组，nodata 位置保持 nodata"""
    mask = (dem == nodata)                 # 无效区掩膜
    dem_work = dem.astype(float)
    dem_work[mask] = np.nan                # 先转成 NaN 方便计算

    # 坡度（弧度）
    grad_y, grad_x = np.gradient(dem_work, cell_size)
    slope = np.sqrt(grad_x**2 + grad_y**2)
    beta = np.arctan(slope)
    beta = np.maximum(beta, tolerance)

    # 汇水面积（高斯平滑演示版）
    contrib = np.ones_like(dem_work)
    contrib[mask] = 0
    flow_acc = gaussian_filter(contrib, sigma=1)
    a = flow_acc * cell_size

    # TWI
    twi = np.full_like(dem_work, np.nan)
    twi = np.log(a / np.tan(beta))
    twi[mask] = nodata          # 把 nodata 填回去
    return twi


# ------------------------------------------------------------------
# 3. 主流程
# ------------------------------------------------------------------
if __name__ == "__main__":
    # 3.1 读入或生成 DEM
    # 这里用 fake_dem 演示；实战时把 dem 换成你自己的数组即可
    df = pd.read_csv('nanling_final_matrix.csv')
    dem = df.to_numpy()
    print("DEM shape:", dem.shape, "min/max %.1f/%.1f m" % (dem.min(), dem.max()))

    # 3.3 计算 TWI
    twi = calculate_twi(dem, nodata=-999)

    # 3.4 绘图
    def nice_imshow(ax, arr, nodata, cmap, label):
        plot_arr = np.ma.masked_where(arr == nodata, arr)
        im = ax.imshow(plot_arr, cmap=cmap, origin='lower')
        plt.colorbar(im, ax=ax, label=label)
        return im

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    nice_imshow(axes[0], dem,  -999, 'terrain', 'Elevation (m)')
    axes[0].set_title('DEM with irregular boundary')

    nice_imshow(axes[1], twi,  -999, 'Blues', 'TWI')
    axes[1].set_title('Topographic Wetness Index')

    plt.tight_layout()
    plt.show()
    pd.DataFrame(twi).to_csv('twi_irregular_matrix.csv')