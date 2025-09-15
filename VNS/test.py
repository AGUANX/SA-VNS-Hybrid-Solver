import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt



def calculate_twi(dem, cell_size=30.0, tolerance=1e-6):
    """
    计算地形湿度指数（TWI）

    参数：
    - dem: 二维 NumPy 数组，数字高程模型
    - cell_size: 每个像元的实际大小（单位：米）
    - tolerance: 防止除零的最小坡度角（弧度）

    返回：
    - twi: 二维 NumPy 数组，地形湿度指数
    """
    # 计算梯度
    grad_y, grad_x = np.gradient(dem, cell_size)
    slope = np.sqrt(grad_x**2 + grad_y**2)
    beta = np.arctan(slope)

    # 防止坡度为0
    beta = np.maximum(beta, tolerance)

    # 计算汇水面积（简化方法：使用高斯平滑模拟）
    # 注意：这不是物理精确的汇水面积，仅用于演示
    # 实际应用中应使用 D8 或 D-infinity 算法
    contributing_area = np.ones_like(dem)
    flow_accumulation = gaussian_filter(contributing_area, sigma=1)
    a = flow_accumulation * cell_size  # 单位面积长度

    # 计算 TWI
    twi = np.log(a / np.tan(beta))

    return twi


def fake_dem(shape=(120, 120), dx=30.0, seed=42):
    """
    生成虚假但具有真实地形特征的 DEM（单位：m）
    """
    rng = np.random.default_rng(seed)
    ny, nx = shape

    # 1) 主坡向：西北 -> 东南
    x, y = np.meshgrid(np.linspace(0, nx-1, nx), np.linspace(0, ny-1, ny))
    main_slope = 0.3 * (x + y)          # 0.3 m/px 综合坡度

    # 2) “V”形谷地（沿对角线）
    valley = 20 * np.exp(-((x - y)**2) / (2 * 15**2))

    # 3) 两个小丘
    hill1 = 40 * np.exp(-((x - 0.3*nx)**2 + (y - 0.7*ny)**2) / (2 * 12**2))
    hill2 = 35 * np.exp(-((x - 0.7*nx)**2 + (y - 0.3*ny)**2) / (2 * 10**2))

    # 4) 高频噪声
    noise = 2.0 * rng.normal(size=shape)

    # 5) 合成
    dem = main_slope - valley + hill1 + hill2 + noise
    dem -= dem.min()          # 确保最低处为 0 m
    return dem

# ==================== 测试 ====================
if __name__ == "__main__":
    dem = fake_dem()
    print("DEM shape:", dem.shape, "min/max %.1f/%.1f m" % (dem.min(), dem.max()))

    plt.figure(figsize=(6, 5))
    im = plt.imshow(dem, cmap='terrain', origin='lower')
    plt.colorbar(im, label='Elevation (m)')
    plt.title("Fake DEM for TWI testing")
    plt.show()

    twi = calculate_twi(dem)

    plt.figure(figsize=(6, 5))
    im = plt.imshow(twi, cmap='terrain', origin='lower')
    plt.colorbar(im, label='Elevation (m)')
    plt.title("Fake DEM for TWI testing")
    plt.show()
