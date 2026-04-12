import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 第一组参数
# ============================================================
weights1 = [0.10733344, 0.22615808, 0.17380262, 0.18810334, 0.13294879, 0.17165373]
means1_rad = [-5.20829490, 10.92689154, -0.02353606, 6.52879456, -2.28892390, 3.90432286]
vars1 = [0.76937016, 1.29324706, 0.89534658, 1.62269598, 0.82791177, 1.67616432]

# ============================================================
# 第二组参数
# ============================================================
weights2 = [0.16066799, 0.15829601, 0.13468776, 0.13684698, 0.21122873, 0.19827252]
means2_rad = [-4.30332511, 11.37027342, 2.51074261, 8.77936079, -0.96909640, 5.40597932]
vars2 = [1.50369282, 0.67148968, 1.29634931, 1.19237576, 1.37704161, 1.22519603]

# ============================================================
# GMM密度计算函数
# ============================================================
def gmm_density(hue_deg, weights, means_rad, variances):
    """计算GMM密度"""
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
# 计算混合后的密度曲线
# ============================================================
print("=" * 60)
print("混合GMM密度函数拟合")
print("=" * 60)

hues = np.linspace(0, 360, 360)
dens1 = [gmm_density(h, weights1, means1_rad, vars1) for h in hues]
dens2 = [gmm_density(h, weights2, means2_rad, vars2) for h in hues]
dens_avg = [(d1 + d2) / 2 for d1, d2 in zip(dens1, dens2)]

# 归一化
max_dens = max(dens_avg)
dens_avg_norm = [d / max_dens for d in dens_avg]

# ============================================================
# 从混合密度曲线重新拟合GMM
# ============================================================
def fit_gmm_from_curve(hues, densities, n_components=8):
    """从密度曲线生成加权样本并拟合GMM"""
    # 根据密度生成加权样本
    weighted_samples = []
    for hue, d in zip(hues, densities):
        # 每个色相根据密度值复制多次
        num_copies = max(1, int(d * 200))
        for _ in range(num_copies):
            weighted_samples.append(hue)
    
    samples_rad = np.deg2rad(weighted_samples).reshape(-1, 1)
    
    # 处理环形：复制偏移±2π的副本
    samples_extended = np.vstack([
        samples_rad,
        samples_rad - 2 * np.pi,
        samples_rad + 2 * np.pi
    ])
    
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(samples_extended)
    
    # 归一化因子
    test_angles = np.linspace(0, 360, 360)
    test_densities = []
    for h in test_angles:
        h_rad = np.deg2rad(h)
        if h_rad > np.pi:
            h_rad -= 2 * np.pi
        elif h_rad < -np.pi:
            h_rad += 2 * np.pi
        log_prob = gmm.score_samples([[h_rad]])
        test_densities.append(np.exp(log_prob)[0])
    
    max_density = max(test_densities)
    
    return gmm, max_density

# ============================================================
# 导出GMM参数
# ============================================================
def export_gmm_params(gmm, max_density, output_file="mapmix/sat_sensitivity_merged_gmm.txt"):
    """导出GMM参数到文本文件"""
    weights = gmm.weights_
    means = gmm.means_.flatten()
    covariances = gmm.covariances_.flatten()
    
    with open(output_file, 'w') as f:
        f.write("# Saturation Sensitivity Merged GMM Parameters\n")
        f.write("# Format: n_components, max_density, weights, means_rad, covariances\n")
        f.write(f"n_components={len(weights)}\n")
        f.write(f"max_density={max_density}\n")
        f.write(f"weights={' '.join(map(str, weights))}\n")
        f.write(f"means_rad={' '.join(map(str, means))}\n")
        f.write(f"covariances={' '.join(map(str, covariances))}\n")
    
    print(f"\nGMM参数已导出到 {output_file}")
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("GMM 参数摘要（可直接用于C++）")
    print("=" * 60)
    print(f"n_components = {len(weights)}")
    print(f"max_density = {max_density:.6f}")
    print("\nweights =", " ".join([f"{w:.8f}" for w in weights]))
    print("\nmeans_rad =", " ".join([f"{m:.8f}" for m in means]))
    print("\ncovariances =", " ".join([f"{c:.8f}" for c in covariances]))
    
    # 转换均值到度数（便于理解）
    print("\n均值转换到度数（0-360°范围）:")
    for i, m in enumerate(means):
        m_deg = np.rad2deg(m) % 360
        print(f"  Component {i}: mean={m_deg:.1f}°, weight={weights[i]:.4f}")

# ============================================================
# 可视化对比
# ============================================================
def visualize_comparison(hues, dens_avg_norm, gmm, max_density):
    """可视化原始混合曲线和GMM拟合结果"""
    # 计算GMM拟合的密度
    gmm_densities = []
    for h in hues:
        h_rad = np.deg2rad(h)
        if h_rad > np.pi:
            h_rad -= 2 * np.pi
        elif h_rad < -np.pi:
            h_rad += 2 * np.pi
        log_prob = gmm.score_samples([[h_rad]])
        gmm_densities.append(np.exp(log_prob)[0] / max_density)
    
    plt.figure(figsize=(14, 6))
    
    # 子图1：混合曲线 vs GMM拟合
    plt.subplot(1, 2, 1)
    plt.plot(hues, dens_avg_norm, 'g-', linewidth=2, label='混合密度曲线')
    plt.plot(hues, gmm_densities, 'r--', linewidth=2, label='GMM拟合')
    plt.xlabel('色相 (度)')
    plt.ylabel('敏感度 (归一化)')
    plt.title('混合曲线 vs GMM拟合')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2：极坐标图
    plt.subplot(1, 2, 2, projection='polar')
    theta_rad = np.deg2rad(hues)
    plt.plot(theta_rad, gmm_densities, 'r-', linewidth=2)
    plt.fill(theta_rad, gmm_densities, alpha=0.3)
    plt.title('饱和度敏感度极坐标图 (GMM拟合)')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('mapmix/sat_sensitivity_merged.png', dpi=150)
    plt.show()

# ============================================================
# 主函数
# ============================================================
def main():
    print("\n正在从混合密度曲线拟合GMM...")
    
    # 拟合GMM（使用8个分量以获得更好的拟合效果）
    gmm, max_density = fit_gmm_from_curve(hues, dens_avg_norm, n_components=8)
    
    # 导出参数
    export_gmm_params(gmm, max_density, "mapmix/sat_sensitivity_merged_gmm.txt")
    
    # 可视化
    visualize_comparison(hues, dens_avg_norm, gmm, max_density)
    
    print("\n" + "=" * 60)
    print("完成！")
    print("输出文件:")
    print("  - mapmix/sat_sensitivity_merged_gmm.txt (GMM参数，用于C++)")
    print("  - mapmix/sat_sensitivity_merged.png (可视化图表)")
    print("=" * 60)

if __name__ == "__main__":
    main()