import numpy as np
import matplotlib.pyplot as plt

# 第一组参数
weights1 = [0.10733344, 0.22615808, 0.17380262, 0.18810334, 0.13294879, 0.17165373]
means1_rad = [-5.20829490, 10.92689154, -0.02353606, 6.52879456, -2.28892390, 3.90432286]
vars1 = [0.76937016, 1.29324706, 0.89534658, 1.62269598, 0.82791177, 1.67616432]

# 第二组参数
weights2 = [0.16066799, 0.15829601, 0.13468776, 0.13684698, 0.21122873, 0.19827252]
means2_rad = [-4.30332511, 11.37027342, 2.51074261, 8.77936079, -0.96909640, 5.40597932]
vars2 = [1.50369282, 0.67148968, 1.29634931, 1.19237576, 1.37704161, 1.22519603]

def gmm_density(hue_deg, weights, means_rad, variances):
    """计算GMM密度"""
    hue_rad = np.deg2rad(hue_deg)
    # 包装到 [-π, π]
    if hue_rad > np.pi: hue_rad -= 2*np.pi
    if hue_rad < -np.pi: hue_rad += 2*np.pi
    
    density = 0.0
    for w, m, v in zip(weights, means_rad, variances):
        diff = hue_rad - m
        exponent = -(diff * diff) / (2.0 * v)
        density += w * np.exp(exponent) / np.sqrt(2.0 * np.pi * v)
    return density

# 计算两组密度并取平均
hues = np.linspace(0, 360, 360)
dens1 = [gmm_density(h, weights1, means1_rad, vars1) for h in hues]
dens2 = [gmm_density(h, weights2, means2_rad, vars2) for h in hues]
dens_avg = [(d1 + d2) / 2 for d1, d2 in zip(dens1, dens2)]

# 归一化
max_dens = max(dens_avg)
dens_avg_norm = [d / max_dens for d in dens_avg]

# 可视化
plt.figure(figsize=(12, 5))
plt.plot(hues, [d/max(dens1) for d in dens1], 'r--', alpha=0.5, label='第一组')
plt.plot(hues, [d/max(dens2) for d in dens2], 'b--', alpha=0.5, label='第二组')
plt.plot(hues, dens_avg_norm, 'g-', linewidth=2, label='平均值')
plt.xlabel('色相 (度)')
plt.ylabel('饱和度敏感度')
plt.title('两组GMM的平均值')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('sat_sensitivity_avg.png', dpi=150)
plt.show()

# 导出平均后的离散点（用于C++查表）
print("平均后的敏感度值（每10度一个点）：")
for hue in range(0, 361, 10):
    # 找到对应索引
    idx = int(hue / 360 * 359)
    print(f"{hue}: {dens_avg_norm[idx]:.4f}")


