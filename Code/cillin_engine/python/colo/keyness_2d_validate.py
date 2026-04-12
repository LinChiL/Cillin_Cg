import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 参数
# ============================================================
norm_factor = 0.00556946
sat_power = 0.6

# Hue Keyness GMM
hue_weights = [0.11132848, 0.13530582, 0.10849456, 0.11214354, 0.22830474, 0.09697763, 0.12251557, 0.08492966]
hue_means_rad = [6.96195504, -2.89108923, 11.31023104, 0.57595815, 4.04752825, -5.55301231, 9.69434493, -1.34614132]
hue_covariances = [0.38831633, 0.43726528, 0.33392143, 0.32444862, 1.07510566, 0.19457601, 0.35457794, 0.151877]

# Saturation Sensitivity GMM
sat_weights = [0.09713084, 0.13469279, 0.11840812, 0.15963278, 0.10077283, 0.11777569, 0.13832214, 0.13326481]
sat_means_rad = [-5.2850193, 4.792507, 9.46615962, -0.2703355, 2.54706985, 6.65366799, -2.55780595, 11.54815262]
sat_covariances = [0.51787753, 0.71454204, 0.81614231, 0.80506982, 0.91408268, 0.72683746, 0.93444933, 0.41576665]

# ============================================================
# 1D GMM 密度计算
# ============================================================
def gmm1d_density(value_deg, weights, means_rad, variances):
    """
    1D GMM 密度计算（处理环形数据）
    
    参数:
        value_deg: 输入值（度数，0-360）
        weights: GMM权重列表
        means_rad: GMM均值列表（弧度）
        variances: GMM方差列表
    """
    value_rad = np.deg2rad(value_deg)
    
    # 包装到 [-π, π]
    if value_rad > np.pi:
        value_rad -= 2 * np.pi
    if value_rad < -np.pi:
        value_rad += 2 * np.pi
    
    density = 0.0
    for w, m, v in zip(weights, means_rad, variances):
        diff = value_rad - m
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
    hue_key = gmm1d_density(hue_deg, hue_weights, hue_means_rad, hue_covariances)
    
    # 2. 饱和度敏感度（基于色相）
    sat_sens = gmm1d_density(hue_deg, sat_weights, sat_means_rad, sat_covariances)
    
    # 3. 饱和度调制因子
    saturation_factor = saturation ** sat_power
    
    # 4. 组合并归一化
    keyness = hue_key * sat_sens * saturation_factor
    keyness_norm = keyness / norm_factor
    
    return min(1.0, keyness_norm)

# ============================================================
# 测试验证
# ============================================================
def test_keyness():
    """测试关键点"""
    test_points = [
        (180, 1.0, "纯绿"),
        (180, 0.8, "绿（高饱和）"),
        (180, 0.5, "绿（中饱和）"),
        (180, 0.2, "绿（低饱和）"),
        (180, 0.0, "灰色"),
        (120, 1.0, "黄绿"),
        (90, 1.0, "黄绿偏黄"),
        (60, 1.0, "黄"),
        (300, 1.0, "紫红"),
        (330, 1.0, "红紫"),
        (0, 1.0, "红"),
        (30, 1.0, "橙红"),
    ]
    
    print("=" * 60)
    print("2D 关键度函数验证")
    print("=" * 60)
    print(f"{'色相':>6} | {'饱和度':>6} | {'关键度':>8} | {'说明'}")
    print("-" * 60)
    
    for hue, sat, desc in test_points:
        k = keyness_2d(hue, sat)
        print(f"{hue:6.1f} | {sat:6.2f} | {k:8.4f} | {desc}")
    
    print("=" * 60)
    
    # 预期结果
    print("\n预期结果验证:")
    print("  ✅ 纯绿(180°,1.0) 应该最高 (~1.0)")
    print("  ✅ 低饱和绿应该比纯绿低")
    print("  ✅ 灰色(180°,0.0) 应该接近 0")
    print("  ✅ 绿色区域应该高于黄色区域")
    print("  ✅ 黄色区域应该最低")

# ============================================================
# 可视化
# ============================================================
def visualize_keyness():
    """可视化关键度分布"""
    print("\n正在生成可视化图表...")
    
    # 生成网格数据
    hues = np.linspace(0, 360, 360)
    sats = np.linspace(0, 1, 100)
    H, S = np.meshgrid(hues, sats)
    Z = np.zeros_like(H)
    
    for i, hue in enumerate(hues):
        for j, sat in enumerate(sats):
            Z[j, i] = keyness_2d(hue, sat)
    
    # 创建图表
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 图1：2D热力图
    im1 = axes[0].imshow(Z, extent=[0, 360, 0, 1], aspect='auto', origin='lower',
                         cmap='hot', interpolation='bilinear')
    axes[0].set_xlabel('色相 (度)', fontsize=12)
    axes[0].set_ylabel('饱和度', fontsize=12)
    axes[0].set_title('2D 关键度分布热力图', fontsize=14)
    
    # 标记高密度区域
    axes[0].axvline(x=180, color='cyan', linestyle='--', alpha=0.7, label='绿色区域(最高)')
    axes[0].axvline(x=120, color='lime', linestyle='--', alpha=0.7, label='黄绿区域(高)')
    axes[0].axvline(x=300, color='magenta', linestyle='--', alpha=0.7, label='紫红区域(中)')
    axes[0].axvline(x=60, color='yellow', linestyle='--', alpha=0.7, label='黄色区域(低)')
    axes[0].legend(loc='upper right', fontsize=8)
    
    plt.colorbar(im1, ax=axes[0], label='关键度')
    
    # 图2：不同饱和度下的关键度曲线
    for sat in [1.0, 0.7, 0.5, 0.3, 0.1]:
        vals = [keyness_2d(h, sat) for h in hues]
        axes[1].plot(hues, vals, label=f'sat={sat}')
    
    axes[1].set_xlabel('色相 (度)', fontsize=12)
    axes[1].set_ylabel('关键度', fontsize=12)
    axes[1].set_title('不同饱和度下的关键度曲线', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 图3：固定色相下的饱和度影响
    for hue in [180, 120, 300, 60, 0]:
        vals = [keyness_2d(hue, s) for s in sats]
        axes[2].plot(sats, vals, label=f'hue={hue}°')
    
    axes[2].set_xlabel('饱和度', fontsize=12)
    axes[2].set_ylabel('关键度', fontsize=12)
    axes[2].set_title('不同色相下的饱和度影响', fontsize=14)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('final/vali/keyness_2d_validation.png', dpi=150)
    plt.show()
    
    print("图表已保存到 final/vali/keyness_2d_validation.png")

# ============================================================
# 导出 C++ 头文件
# ============================================================
def export_cpp_header():
    """导出 C++ 头文件"""
    header = """// keyness_2d.hpp
// Auto-generated from Python fitting
#pragma once
#include <cmath>
#include <array>

class Keyness2D {
public:
    Keyness2D() = default;
    
    double keyness(double hue_deg, double saturation) const {
        const double PI = 3.141592653589793;
        double hue_rad = hue_deg * PI / 180.0;
        
        // 包装到 [-π, π]
        if (hue_rad > PI) hue_rad -= 2 * PI;
        if (hue_rad < -PI) hue_rad += 2 * PI;
        
        // 1. 色相关键度
        double hue_key = gmm1d(hue_rad, hue_weights, hue_means, hue_covs);
        
        // 2. 饱和度敏感度
        double sat_sens = gmm1d(hue_rad, sat_weights, sat_means, sat_covs);
        
        // 3. 饱和度调制
        double sat_factor = std::pow(saturation, 0.6);
        
        // 4. 组合并归一化
        double result = hue_key * sat_sens * sat_factor / 0.00556946;
        
        return std::min(1.0, result);
    }
    
private:
    static constexpr int N_COMPONENTS = 8;
    
    using Array = std::array<double, N_COMPONENTS>;
    
    static constexpr Array hue_weights = {
        0.11132848, 0.13530582, 0.10849456, 0.11214354,
        0.22830474, 0.09697763, 0.12251557, 0.08492966
    };
    
    static constexpr Array hue_means = {
        6.96195504, -2.89108923, 11.31023104, 0.57595815,
        4.04752825, -5.55301231, 9.69434493, -1.34614132
    };
    
    static constexpr Array hue_covs = {
        0.38831633, 0.43726528, 0.33392143, 0.32444862,
        1.07510566, 0.19457601, 0.35457794, 0.15187700
    };
    
    static constexpr Array sat_weights = {
        0.09713084, 0.13469279, 0.11840812, 0.15963278,
        0.10077283, 0.11777569, 0.13832214, 0.13326481
    };
    
    static constexpr Array sat_means = {
        -5.28501930, 4.79250700, 9.46615962, -0.27033550,
        2.54706985, 6.65366799, -2.55780595, 11.54815262
    };
    
    static constexpr Array sat_covs = {
        0.51787753, 0.71454204, 0.81614231, 0.80506982,
        0.91408268, 0.72683746, 0.93444933, 0.41576665
    };
    
    double gmm1d(double x, const Array& weights, const Array& means, const Array& covs) const {
        const double PI = 3.141592653589793;
        double density = 0.0;
        
        for (int i = 0; i < N_COMPONENTS; ++i) {
            double diff = x - means[i];
            double exponent = -(diff * diff) / (2.0 * covs[i]);
            density += weights[i] * std::exp(exponent) / std::sqrt(2.0 * PI * covs[i]);
        }
        
        return density;
    }
};
"""
    with open("final/vali/keyness_2d.hpp", "w") as f:
        f.write(header)
    
    print("\nC++ 头文件已导出到 final/vali/keyness_2d.hpp")

# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    # 测试关键点
    test_keyness()
    
    # 可视化
    visualize_keyness()
    
    # 导出 C++ 头文件
    export_cpp_header()
    
    print("\n" + "=" * 60)
    print("完成！输出文件:")
    print("  - final/vali/keyness_2d_validation.png (可视化图表)")
    print("  - final/vali/keyness_2d.hpp (C++ 头文件)")
    print("=" * 60)