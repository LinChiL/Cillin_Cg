// keyness_2d.hpp
// 计算颜色感知敏感度权重
// 值越高 = 人眼越敏感 = 需要越多颜色锚点

#pragma once
#include <cmath>
#include <array>

class Keyness2D {
public:
    Keyness2D() = default;
    
    // 输入: 色相(0-360°), 饱和度(0-1)
    // 输出: 敏感度权重(0-1)
    double keyness(double hue_deg, double saturation) const {
        const double PI = 3.141592653589793;
        double hue_rad = hue_deg * PI / 180.0;
        
        // 包装到 [-PI, PI]
        if (hue_rad > PI) hue_rad -= 2 * PI;
        if (hue_rad < -PI) hue_rad += 2 * PI;
        
        double hue_key = gmm1d(hue_rad, hue_weights, hue_means, hue_covs);
        double sat_sens = gmm1d(hue_rad, sat_weights, sat_means, sat_covs);
        double sat_factor = std::pow(saturation, 0.6);
        
        double result = hue_key * sat_sens * sat_factor / 0.00556946;
        
        return std::min(1.0, result);
    }
    
private:
    static constexpr int N = 8;
    using Array = std::array<double, N>;
    
    // 色相敏感度 GMM (从锚点密度拟合)
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
    
    // 饱和度敏感度 GMM (从实验数据拟合)
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
    
    double gmm1d(double x, const Array& w, const Array& m, const Array& v) const {
        const double PI = 3.141592653589793;
        double density = 0.0;
        
        for (int i = 0; i < N; ++i) {
            double diff = x - m[i];
            double exp_term = -(diff * diff) / (2.0 * v[i]);
            density += w[i] * std::exp(exp_term) / std::sqrt(2.0 * PI * v[i]);
        }
        
        return density;
    }
};