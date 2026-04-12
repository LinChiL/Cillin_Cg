import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 读取饱和度敏感度数据
# ============================================================
def load_saturation_data(filename="saturation_sensitivity.txt"):
    """读取 txt 文件，格式：色相:阈值"""
    hues = []
    thresholds = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # 支持两种分隔符：冒号或空格
            if ':' in line:
                parts = line.split(':')
            else:
                parts = line.split()
            
            if len(parts) >= 2:
                hue = float(parts[0])
                threshold = float(parts[1])
                hues.append(hue)
                thresholds.append(threshold)
    
    # 转换为 numpy 数组
    hues = np.array(hues)
    thresholds = np.array(thresholds)
    
    # 按色相排序
    sort_idx = np.argsort(hues)
    hues = hues[sort_idx]
    thresholds = thresholds[sort_idx]
    
    # 添加首尾连接点（处理环形）
    if hues[0] > 0:
        hues = np.concatenate([[0.0], hues])
        thresholds = np.concatenate([[thresholds[0]], thresholds])
    if hues[-1] < 360:
        hues = np.concatenate([hues, [360.0]])
        thresholds = np.concatenate([thresholds, [thresholds[0]]])
    
    return hues, thresholds

# ============================================================
# 将阈值转换为敏感度
# ============================================================
def threshold_to_sensitivity(thresholds):
    """阈值越小，敏感度越高"""
    # 敏感度 = 1 / 阈值（归一化后）
    sensitivity = 1.0 / np.maximum(thresholds, 0.01)
    # 归一化到 0-1
    sensitivity = sensitivity / np.max(sensitivity)
    return sensitivity

# ============================================================
# 方法1：高斯核密度估计（KDE）
# ============================================================
def fit_kde(hues, sensitivities, sigma=5.0):
    """使用高斯核拟合敏感度曲线"""
    def sensitivity_func(theta_deg):
        total = 0.0
        for h, s in zip(hues, sensitivities):
            diff = min(abs(theta_deg - h), 360 - abs(theta_deg - h))
            total += s * np.exp(-(diff * diff) / (2 * sigma * sigma))
        return total
    
    # 归一化
    test_angles = np.linspace(0, 360, 360)
    test_vals = np.array([sensitivity_func(a) for a in test_angles])
    max_val = np.max(test_vals)
    
    def normalized_func(theta_deg):
        return sensitivity_func(theta_deg) / max_val
    
    return normalized_func

# ============================================================
# 方法2：高斯混合模型（GMM）- 推荐
# ============================================================
def fit_gmm(hues, sensitivities, n_components=6):
    """使用GMM拟合敏感度曲线"""
    # 将敏感度作为权重，生成加权样本点
    weighted_samples = []
    for h, w in zip(hues, sensitivities):
        # 每个锚点根据敏感度权重复制多次
        num_copies = max(1, int(w * 100))
        for _ in range(num_copies):
            weighted_samples.append(h)
    
    samples_rad = np.deg2rad(weighted_samples).reshape(-1, 1)
    
    # 处理环形：复制偏移±2π的副本
    samples_extended = np.vstack([
        samples_rad,
        samples_rad - 2*np.pi,
        samples_rad + 2*np.pi
    ])
    
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(samples_extended)
    
    def density(theta_deg):
        theta_rad = np.deg2rad(theta_deg)
        # 包装到 [-π, π]
        if theta_rad > np.pi:
            theta_rad -= 2*np.pi
        elif theta_rad < -np.pi:
            theta_rad += 2*np.pi
        
        log_prob = gmm.score_samples([[theta_rad]])
        return np.exp(log_prob)[0]
    
    # 归一化
    test_angles = np.linspace(0, 360, 360)
    test_densities = [density(a) for a in test_angles]
    max_density = max(test_densities)
    
    def normalized_func(theta_deg):
        return density(theta_deg) / max_density
    
    return normalized_func, gmm

# ============================================================
# 方法3：样条插值（简单平滑）
# ============================================================
def fit_spline(hues, sensitivities, smooth=0.05):
    """使用样条插值拟合敏感度曲线"""
    # 复制一份处理环形
    hues_extended = np.concatenate([hues - 360, hues, hues + 360])
    sens_extended = np.concatenate([sensitivities, sensitivities, sensitivities])
    
    from scipy.interpolate import UnivariateSpline
    spline = UnivariateSpline(hues_extended, sens_extended, s=smooth)
    
    def func(theta_deg):
        return spline(theta_deg)
    
    # 归一化
    test_vals = func(np.linspace(0, 360, 360))
    max_val = np.max(test_vals)
    
    def normalized_func(theta_deg):
        return func(theta_deg) / max_val
    
    return normalized_func

# ============================================================
# 导出参数供C++使用
# ============================================================
def export_gmm_params(gmm, output_file="sat_sensitivity_gmm.txt"):
    """导出GMM参数到文本文件"""
    means = gmm.means_.flatten()
    covariances = gmm.covariances_.flatten()
    weights = gmm.weights_
    
    with open(output_file, 'w') as f:
        f.write("# Saturation Sensitivity GMM Parameters\n")
        f.write(f"# Format: n_components, weights, means_rad, covariances\n")
        f.write(f"n_components={len(weights)}\n")
        f.write(f"weights={' '.join(map(str, weights))}\n")
        f.write(f"means_rad={' '.join(map(str, means))}\n")
        f.write(f"covariances={' '.join(map(str, covariances))}\n")
    
    print(f"GMM parameters exported to {output_file}")
    
    # 打印摘要
    print("\n" + "="*60)
    print("GMM 参数摘要")
    print("="*60)
    for i, (w, m, c) in enumerate(zip(weights, means, covariances)):
        m_deg = np.rad2deg(m) % 360
        print(f"  Component {i}: weight={w:.4f}, mean={m_deg:.1f}°, variance={c:.6f}")

# ============================================================
# 可视化
# ============================================================
def visualize(hues, thresholds, sensitivities, kde_func, gmm_func, spline_func):
    """对比三种拟合方法"""
    theta_range = np.linspace(0, 360, 720)
    
    kde_vals = np.array([kde_func(t) for t in theta_range])
    gmm_vals = np.array([gmm_func(t) for t in theta_range])
    spline_vals = np.array([spline_func(t) for t in theta_range])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 子图1: 原始数据（阈值）
    ax1 = axes[0, 0]
    ax1.scatter(hues, thresholds, c='red', s=50, zorder=5)
    ax1.plot(hues, thresholds, 'r--', alpha=0.5)
    ax1.set_xlabel('色相 (度)')
    ax1.set_ylabel('饱和度微分阈值')
    ax1.set_title('原始实验数据：饱和度微分阈值')
    ax1.set_xlim(0, 360)
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 敏感度转换
    ax2 = axes[0, 1]
    ax2.scatter(hues, sensitivities, c='green', s=50, zorder=5)
    ax2.plot(hues, sensitivities, 'g--', alpha=0.5)
    ax2.set_xlabel('色相 (度)')
    ax2.set_ylabel('敏感度 (归一化)')
    ax2.set_title('转换后：饱和度敏感度')
    ax2.set_xlim(0, 360)
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 三种拟合方法对比
    ax3 = axes[1, 0]
    ax3.plot(theta_range, kde_vals, 'b-', linewidth=1.5, label='KDE')
    ax3.plot(theta_range, gmm_vals, 'g-', linewidth=1.5, label='GMM')
    ax3.plot(theta_range, spline_vals, 'r-', linewidth=1.5, label='样条插值')
    ax3.scatter(hues, sensitivities, c='black', s=30, zorder=5, label='原始数据')
    ax3.set_xlabel('色相 (度)')
    ax3.set_ylabel('敏感度 (归一化)')
    ax3.set_title('三种拟合方法对比')
    ax3.set_xlim(0, 360)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 子图4: 敏感度分布热力图（极坐标）
    ax4 = axes[1, 1]
    ax4 = plt.subplot(2, 2, 4, projection='polar')
    theta_rad = np.deg2rad(theta_range)
    ax4.plot(theta_rad, gmm_vals, 'g-', linewidth=2)
    ax4.fill(theta_rad, gmm_vals, alpha=0.3)
    ax4.set_title('饱和度敏感度极坐标图')
    ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('map2/saturation_sensitivity.png', dpi=150)
    plt.show()

# ============================================================
# 主函数
# ============================================================
def main():
    print("="*60)
    print("饱和度敏感度函数拟合")
    print("="*60)
    
    # 读取数据
    try:
        hues, thresholds = load_saturation_data("../../html/ColMap2.txt")
    except FileNotFoundError:
        print("错误：找不到 ColMap2.txt 文件")
        print("请确保文件格式为：")
        print("0.0:0.15")
        print("30.0:0.25")
        print("...")
        return
    
    print(f"读取数据点: {len(hues)} 个")
    print(f"色相范围: {hues[0]:.1f}° - {hues[-1]:.1f}°")
    print(f"阈值范围: {np.min(thresholds):.4f} - {np.max(thresholds):.4f}")
    
    # 转换为敏感度
    sensitivities = threshold_to_sensitivity(thresholds)
    
    # 拟合
    print("\n正在拟合...")
    kde_func = fit_kde(hues, sensitivities, sigma=8.0)
    gmm_func, gmm = fit_gmm(hues, sensitivities, n_components=6)
    spline_func = fit_spline(hues, sensitivities, smooth=0.01)
    
    # 可视化
    visualize(hues, thresholds, sensitivities, kde_func, gmm_func, spline_func)
    
    # 导出GMM参数
    export_gmm_params(gmm, "map2/sat_sensitivity_gmm.txt")
    
    # 生成采样建议
    print("\n" + "="*60)
    print("饱和度敏感度分析结果")
    print("="*60)
    
    # 找出敏感度最高的色相
    test_angles = np.linspace(0, 360, 360)
    test_sens = [gmm_func(a) for a in test_angles]
    max_idx = np.argmax(test_sens)
    print(f"最高敏感度色相: {test_angles[max_idx]:.1f}°")
    
    # 找出敏感度最低的色相
    min_idx = np.argmin(test_sens)
    print(f"最低敏感度色相: {test_angles[min_idx]:.1f}°")
    
    print("\n拟合完成！")
    print("输出文件:")
    print("  - saturation_sensitivity.png (可视化图表)")
    print("  - sat_sensitivity_gmm.txt (GMM参数，可用于C++)")

if __name__ == "__main__":
    main()