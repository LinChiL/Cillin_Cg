import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.signal import savgol_filter

# ============================================================
# 你的锚定色相数据（从RGB数据中提取）
# ============================================================
anchor_angles = [
    0.00, 8.64, 13.92, 16.80, 19.68, 22.56, 25.44, 28.32, 30.72, 33.60,
    36.48, 39.36, 42.24, 45.12, 48.00, 50.40, 53.28, 56.16, 61.92, 70.56,
    78.72, 90.24, 104.16, 112.32, 126.72, 134.88, 143.52, 154.56, 166.08,
    171.36, 174.24, 177.12, 180.00, 182.88, 185.76, 188.64, 191.04, 193.92,
    196.80, 199.68, 202.56, 205.44, 208.32, 210.72, 213.60, 216.48, 219.36,
    222.24, 225.12, 228.00, 230.40, 236.16, 241.92, 250.08, 255.84, 258.72,
    261.60, 264.48, 267.36, 269.76, 272.64, 275.52, 278.40, 281.28, 284.16,
    287.04, 289.44, 292.32, 295.20, 298.08, 300.96, 306.72, 312.00, 320.64,
    329.28, 337.44, 348.96, 354.24
]

# ============================================================
# 方法1：高斯核密度估计（KDE）
# ============================================================
def gaussian_kde(angles, sigma=3.0):
    """使用高斯核估计关键度密度函数"""
    angles_rad = np.deg2rad(angles)
    
    def density(theta_deg):
        theta_rad = np.deg2rad(theta_deg)
        total = 0.0
        for a in angles_rad:
            diff = abs(theta_rad - a)
            diff = min(diff, 2*np.pi - diff)  # 环形距离
            total += np.exp(-(diff * diff) / (2 * (np.deg2rad(sigma))**2))
        return total / (len(angles) * np.sqrt(2*np.pi) * np.deg2rad(sigma))
    
    return density

# ============================================================
# 方法2：高斯混合模型（GMM）- 更精确
# ============================================================
def fit_gmm(angles, n_components=6):
    """使用GMM拟合关键度分布"""
    angles_rad = np.deg2rad(angles).reshape(-1, 1)
    
    # 处理环形数据：复制一份偏移±2π的副本
    angles_extended = np.vstack([
        angles_rad,
        angles_rad + 2*np.pi,
        angles_rad - 2*np.pi
    ])
    
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(angles_extended)
    
    def density(theta_deg):
        theta_rad = np.deg2rad(theta_deg)
        # 包装到 [-π, π] 范围
        theta_wrapped = theta_rad
        if theta_wrapped > np.pi:
            theta_wrapped -= 2*np.pi
        elif theta_wrapped < -np.pi:
            theta_wrapped += 2*np.pi
        
        # 计算GMM概率密度
        log_prob = gmm.score_samples([[theta_wrapped]])
        return np.exp(log_prob)[0]
    
    # 归一化使最大值为1
    test_angles = np.linspace(0, 360, 360)
    test_densities = [density(a) for a in test_angles]
    max_density = max(test_densities)
    
    def normalized_density(theta_deg):
        return density(theta_deg) / max_density
    
    return normalized_density, gmm

# ============================================================
# 方法3：基于相邻点距离的局部密度（最直接）
# ============================================================
def local_density_from_gaps(angles, smooth_window=5):
    """基于相邻锚点间距计算局部密度"""
    sorted_angles = sorted(angles)
    n = len(sorted_angles)
    
    # 计算相邻点间距（考虑环形）
    gaps = []
    for i in range(n):
        next_i = (i + 1) % n
        gap = sorted_angles[next_i] - sorted_angles[i]
        if gap < 0:
            gap += 360
        gaps.append(gap)
    
    # 密度 = 1 / 间距（归一化）
    densities = [1.0 / max(g, 0.1) for g in gaps]
    
    # 将密度值赋给每个锚点
    point_densities = {}
    for i, angle in enumerate(sorted_angles):
        point_densities[angle] = densities[i]
    
    def density_func(theta_deg):
        # 找到最近的锚点，使用其密度
        min_dist = 360
        closest_density = 0
        for a, d in point_densities.items():
            dist = min(abs(theta_deg - a), 360 - abs(theta_deg - a))
            if dist < min_dist:
                min_dist = dist
                closest_density = d
        return closest_density
    
    return density_func

# ============================================================
# 可视化对比
# ============================================================
def visualize_density_functions(anchor_angles):
    """对比三种密度估计方法"""
    theta_range = np.linspace(0, 360, 720)
    
    # 方法1: 高斯KDE
    kde_density = gaussian_kde(anchor_angles, sigma=5.0)
    kde_values = [kde_density(t) for t in theta_range]
    
    # 方法2: GMM
    gmm_density, _ = fit_gmm(anchor_angles, n_components=8)
    gmm_values = [gmm_density(t) for t in theta_range]
    
    # 方法3: 局部密度
    local_density = local_density_from_gaps(anchor_angles)
    local_values = [local_density(t) for t in theta_range]
    
    # 绘制对比图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 子图1: 锚点分布（柱状图）
    ax1 = axes[0, 0]
    ax1.hist(anchor_angles, bins=36, range=(0, 360), edgecolor='black', alpha=0.7)
    ax1.set_xlabel('色相角 (度)')
    ax1.set_ylabel('锚点数量')
    ax1.set_title('锚定色相分布直方图')
    ax1.set_xlim(0, 360)
    
    # 子图2: 高斯KDE结果
    ax2 = axes[0, 1]
    ax2.plot(theta_range, kde_values, 'b-', linewidth=2, label='高斯KDE')
    ax2.fill_between(theta_range, 0, kde_values, alpha=0.3)
    ax2.scatter(anchor_angles, [0.01]*len(anchor_angles), c='red', s=20, alpha=0.5, label='锚点')
    ax2.set_xlabel('色相角 (度)')
    ax2.set_ylabel('关键度密度')
    ax2.set_title('高斯核密度估计 (KDE)')
    ax2.set_xlim(0, 360)
    ax2.legend()
    
    # 子图3: GMM结果
    ax3 = axes[1, 0]
    ax3.plot(theta_range, gmm_values, 'g-', linewidth=2, label='GMM')
    ax3.fill_between(theta_range, 0, gmm_values, alpha=0.3)
    ax3.scatter(anchor_angles, [0.01]*len(anchor_angles), c='red', s=20, alpha=0.5, label='锚点')
    ax3.set_xlabel('色相角 (度)')
    ax3.set_ylabel('关键度 (归一化)')
    ax3.set_title('高斯混合模型 (GMM)')
    ax3.set_xlim(0, 360)
    ax3.legend()
    
    # 子图4: 三种方法对比
    ax4 = axes[1, 1]
    ax4.plot(theta_range, [v / max(kde_values) for v in kde_values], 'b--', linewidth=1.5, label='KDE')
    ax4.plot(theta_range, gmm_values, 'g-', linewidth=1.5, label='GMM')
    ax4.plot(theta_range, [v / max(local_values) for v in local_values], 'r:', linewidth=1.5, label='局部密度')
    ax4.set_xlabel('色相角 (度)')
    ax4.set_ylabel('关键度 (归一化)')
    ax4.set_title('三种方法对比')
    ax4.set_xlim(0, 360)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('keyness_density_comparison.png', dpi=150)
    plt.show()
    
    return kde_values, gmm_values, local_values

# ============================================================
# 导出GMM参数供C++使用
# ============================================================
def export_gmm_params(gmm, output_file="gmm_params.txt"):
    """导出GMM参数到文本文件，便于在C++中实现"""
    means = gmm.means_.flatten()
    covariances = gmm.covariances_.flatten()
    weights = gmm.weights_
    
    with open(output_file, 'w') as f:
        f.write(f"# GMM Parameters for Hue Keyness Function\n")
        f.write(f"# Format: n_components, weights, means, covariances\n")
        f.write(f"n_components={len(weights)}\n")
        f.write(f"weights={' '.join(map(str, weights))}\n")
        f.write(f"means_rad={' '.join(map(str, means))}\n")
        f.write(f"covariances={' '.join(map(str, covariances))}\n")
    
    print(f"GMM parameters exported to {output_file}")
    
    # 打印到控制台
    print("\n" + "="*60)
    print("GMM 参数（可直接用于C++实现）")
    print("="*60)
    print(f"components: {len(weights)}")
    for i, (w, m, c) in enumerate(zip(weights, means, covariances)):
        print(f"  {i}: weight={w:.4f}, mean={m:.4f} rad, cov={c:.6f}")

# ============================================================
# 生成采样建议
# ============================================================
def generate_sampling_schedule(density_func, total_samples=96):
    """根据关键度密度函数生成采样计划"""
    theta_range = np.linspace(0, 360, 3600)
    densities = np.array([density_func(t) for t in theta_range])
    
    # 根据密度分配采样点
    cumulative = np.cumsum(densities)
    cumulative = cumulative / cumulative[-1]
    
    samples = []
    for i in range(total_samples):
        target = i / total_samples
        idx = np.searchsorted(cumulative, target)
        samples.append(theta_range[idx])
    
    return samples

# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    print("="*60)
    print("色相关键度密度函数拟合")
    print("="*60)
    print(f"锚定色相数量: {len(anchor_angles)}")
    print(f"色相范围: {min(anchor_angles):.2f}° - {max(anchor_angles):.2f}°")
    
    # 可视化对比
    kde_vals, gmm_vals, local_vals = visualize_density_functions(anchor_angles)
    
    # 拟合GMM并导出参数
    gmm_density, gmm_model = fit_gmm(anchor_angles, n_components=8)
    export_gmm_params(gmm_model, "gmm_params.txt")
    
    # 生成96色的采样建议
    print("\n" + "="*60)
    print("基于关键度密度的96色采样建议")
    print("="*60)
    
    samples = generate_sampling_schedule(gmm_density, total_samples=96)
    
    # 打印采样结果
    print("\n推荐采样色相（按关键度密度分配）:")
    for i, hue in enumerate(samples):
        if i % 8 == 0:
            print()
        print(f"{hue:6.2f}°", end="")
    print("\n")
    
    # 保存采样结果
    with open("sampled_hues.txt", "w") as f:
        for hue in samples:
            f.write(f"{hue:.2f}\n")
    
    print("采样结果已保存到 sampled_hues.txt")
    print("GMM参数已保存到 gmm_params.txt")