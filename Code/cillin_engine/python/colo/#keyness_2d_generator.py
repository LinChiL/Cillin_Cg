import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 色相关键度 GMM 参数
# ============================================================
hue_weights = [0.11132848, 0.13530582, 0.10849456, 0.11214354, 0.22830474, 0.09697763, 0.12251557, 0.08492966]
hue_means_rad = [6.96195504, -2.89108923, 11.31023104, 0.57595815, 4.04752825, -5.55301231, 9.69434493, -1.34614132]
hue_vars = [0.38831633, 0.43726528, 0.33392143, 0.32444862, 1.07510566, 0.19457601, 0.35457794, 0.15187700]

# ============================================================
# 饱和度敏感度 GMM 参数（混合后的）
# ============================================================
sat_weights = [0.09713084, 0.13469279, 0.11840812, 0.15963278, 0.10077283, 0.11777569, 0.13832214, 0.13326481]
sat_means_rad = [-5.28501930, 4.79250700, 9.46615962, -0.27033550, 2.54706985, 6.65366799, -2.55780595, 11.54815262]
sat_vars = [0.51787753, 0.71454204, 0.81614231, 0.80506982, 0.91408268, 0.72683746, 0.93444933, 0.41576665]

# ============================================================
# 1D GMM 密度计算
# ============================================================
def gmm1d_density(hue_deg, weights, means_rad, variances):
    """1D GMM 密度计算"""
    hue_rad = np.deg2rad(hue_deg)
    # 包装到 [-π, π]
    if hue_rad > np.pi:
        hue_rad -= 2 * np.pi
    if hue_rad < -np.pi:
        hue_rad += 2 * np.pi
    
    density = 0.0
    for w, m, v in zip(weights, means_rad, variances):
        diff = hue_rad - m
        exponent = -(diff * diff) / (2.0 * v)
        density += w * np.exp(exponent) / np.sqrt(2.0 * np.pi * v)
    return density

# ============================================================
# 2D 关键度函数
# ============================================================
def keyness_2d(hue_deg, saturation):
    """
    计算颜色的关键度（采样密度权重）
    
    参数:
        hue_deg: 色相 (0-360°)
        saturation: 饱和度 (0-1)
    
    返回:
        关键度 (0-1)，越高表示该区域需要更多采样点
    """
    # 1. 色相关键度
    hue_key = gmm1d_density(hue_deg, hue_weights, hue_means_rad, hue_vars)
    
    # 2. 饱和度敏感度（基于是色相）
    sat_sens = gmm1d_density(hue_deg, sat_weights, sat_means_rad, sat_vars)
    
    # 3. 饱和度调制因子
    # 饱和度越高，越接近纯色，需要更高密度
    # 饱和度越低（灰色），密度降低
    saturation_factor = saturation ** 0.6  # 可调参数
    
    # 4. 组合
    keyness = hue_key * sat_sens * saturation_factor
    
    return keyness

# ============================================================
# 归一化关键度函数（输出 0-1）
# ============================================================
def normalize_keyness_2d():
    """计算归一化因子"""
    max_val = 0.0
    for hue in range(0, 360, 10):
        for sat in np.linspace(0, 1, 11):
            val = keyness_2d(hue, sat)
            if val > max_val:
                max_val = val
    return max_val

# ============================================================
# 可视化
# ============================================================
def visualize_2d_keyness():
    """可视化2D关键度分布"""
    print("正在计算2D关键度分布...")
    
    # 计算归一化因子
    norm_factor = normalize_keyness_2d()
    print(f"归一化因子: {norm_factor:.6f}")
    
    # 生成网格数据
    hues = np.linspace(0, 360, 180)
    sats = np.linspace(0, 1, 50)
    H, S = np.meshgrid(hues, sats)
    Z = np.zeros_like(H)
    
    for i in range(len(hues)):
        for j in range(len(sats)):
            Z[j, i] = keyness_2d(hues[i], sats[j]) / norm_factor
    
    # 图1：2D热力图
    fig = plt.figure(figsize=(16, 8))
    
    ax1 = fig.add_subplot(121)
    im = ax1.imshow(Z, extent=[0, 360, 0, 1], aspect='auto', origin='lower',
                    cmap='hot', interpolation='bilinear')
    ax1.set_xlabel('色相 (度)', fontsize=12)
    ax1.set_ylabel('饱和度', fontsize=12)
    ax1.set_title('2D 关键度分布热力图', fontsize=14)
    plt.colorbar(im, ax=ax1, label='关键度')
    
    # 标记高密度区域
    ax1.axvline(x=180, color='cyan', linestyle='--', alpha=0.5, label='绿色区域(高)')
    ax1.axvline(x=300, color='magenta', linestyle='--', alpha=0.5, label='紫红区域(中)')
    ax1.legend()
    
    # 图2：3D曲面图
    ax2 = fig.add_subplot(122, projection='3d')
    surf = ax2.plot_surface(H, S, Z, cmap='hot', alpha=0.8, linewidth=0, antialiased=True)
    ax2.set_xlabel('色相 (度)', fontsize=10)
    ax2.set_ylabel('饱和度', fontsize=10)
    ax2.set_zlabel('关键度', fontsize=10)
    ax2.set_title('2D 关键度分布曲面', fontsize=14)
    ax2.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.savefig('final/keyness_2d_distribution.png', dpi=150)
    plt.show()
    
    return norm_factor

# ============================================================
# 导出采样点
# ============================================================
def export_sampling_points(norm_factor, num_hue=24, num_sat=8):
    """导出采样点（用于颜色库生成）"""
    print("\n" + "=" * 60)
    print("基于关键度的采样建议")
    print("=" * 60)
    
    # 计算累积分布
    hues = np.linspace(0, 360, 360)
    sat_levels = np.linspace(0, 1, 20)
    
    # 色相方向的累积分布
    hue_keyness = [keyness_2d(h, 1.0) / norm_factor for h in hues]
    hue_cdf = np.cumsum(hue_keyness)
    hue_cdf = hue_cdf / hue_cdf[-1]
    
    # 采样色相
    sampled_hues = []
    for i in range(num_hue):
        target = (i + 0.5) / num_hue
        idx = np.searchsorted(hue_cdf, target)
        sampled_hues.append(hues[idx])
    
    print("\n推荐的色相采样点:")
    for i, h in enumerate(sampled_hues):
        print(f"  {i+1:2d}: {h:6.2f}°")
    
    # 对于每个色相，采样饱和度
    print("\n推荐的饱和度采样点（针对每个色相）:")
    for h in sampled_hues[:5]:  # 只显示前5个作为示例
        sat_keyness = [keyness_2d(h, s) / norm_factor for s in sat_levels]
        sat_cdf = np.cumsum(sat_keyness)
        sat_cdf = sat_cdf / sat_cdf[-1]
        
        sampled_sats = []
        for i in range(num_sat):
            target = (i + 0.5) / num_sat
            idx = np.searchsorted(sat_cdf, target)
            sampled_sats.append(sat_levels[idx])
        
        print(f"\n  色相 {h:.1f}° 的饱和度采样:")
        for s in sampled_sats:
            print(f"    {s:.3f}")

# ============================================================
# 导出C++参数
# ============================================================
def export_cpp_params(norm_factor):
    """导出C++需要的参数"""
    with open("final/keyness_2d_params.txt", "w") as f:
        f.write("# 2D Keyness Function Parameters\n")
        f.write("# Format: norm_factor, hue_gmm_params, sat_gmm_params\n\n")
        f.write(f"norm_factor = {norm_factor:.8f}\n\n")
        
        f.write("# Hue Keyness GMM\n")
        f.write(f"hue_n_components = {len(hue_weights)}\n")
        f.write(f"hue_weights = {' '.join(map(str, hue_weights))}\n")
        f.write(f"hue_means_rad = {' '.join(map(str, hue_means_rad))}\n")
        f.write(f"hue_covariances = {' '.join(map(str, hue_vars))}\n\n")
        
        f.write("# Saturation Sensitivity GMM\n")
        f.write(f"sat_n_components = {len(sat_weights)}\n")
        f.write(f"sat_weights = {' '.join(map(str, sat_weights))}\n")
        f.write(f"sat_means_rad = {' '.join(map(str, sat_means_rad))}\n")
        f.write(f"sat_covariances = {' '.join(map(str, sat_vars))}\n\n")
        
        f.write("# Saturation Modulation Factor\n")
        f.write("sat_power = 0.6\n")
    
    print("\nC++参数已导出到 keyness_2d_params.txt")

# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("2D 关键度函数生成器")
    print("=" * 60)
    
    # 可视化
    norm_factor = visualize_2d_keyness()
    
    # 导出采样点
    export_sampling_points(norm_factor)
    
    # 导出C++参数
    export_cpp_params(norm_factor)
    
    print("\n" + "=" * 60)
    print("完成！输出文件:")
    print("  - final/keyness_2d_distribution.png (2D分布图)")
    print("  - final/keyness_2d_params.txt (C++参数)")
    print("=" * 60)