#define CGLTF_IMPLEMENTATION
#include "cgltf.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <vector>
#include <array>
#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <map>
#include <cstdint>
#include <cstring>
#include <random>

// --- 基础数学结构 ---
struct Vec3 {
    float x, y, z;
    Vec3 operator-(const Vec3& v) const { return {x - v.x, y - v.y, z - v.z}; }
    Vec3 operator+(const Vec3& v) const { return {x + v.x, y + v.y, z + v.z}; }
    Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    float dot(const Vec3& v) const { return x * v.x + y * v.y + z * v.z; }
    Vec3 cross(const Vec3& v) const { return {y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x}; }
    float length_sq() const { return x * x + y * y + z * z; }
    float length() const { return std::sqrt(length_sq()); }
};

struct Color { uint8_t r, g, b; };

struct Vec2 { float u, v; };

struct HSV {
    double h; // 0-360
    double s; // 0-1
    double v; // 0-1
};

// --- Keyness2D: 人眼感知敏感度权重计算 (GMM) ---
class Keyness2D {
public:
    double keyness(double hue_deg, double saturation) const {
        const double PI = 3.141592653589793;
        double hue_rad = hue_deg * PI / 180.0;
        
        if (hue_rad > PI) hue_rad -= 2 * PI;
        if (hue_rad < -PI) hue_rad += 2 * PI;
        
        double hue_key = gmm1d(hue_rad, hue_weights, hue_means, hue_covs);
        double sat_sens = gmm1d(hue_rad, sat_weights, sat_means, sat_covs);
        double sat_factor = std::pow(saturation, 0.6);
        
        // 归一化系数：0.00556946 是实验测定的峰值密度
        double result = hue_key * sat_sens * sat_factor / 0.00556946;
        return std::min(1.0, result);
    }

private:
    static constexpr int N = 8;
    typedef std::array<double, N> Array;

    static constexpr Array hue_weights = {0.11132848, 0.13530582, 0.10849456, 0.11214354, 0.22830474, 0.09697763, 0.12251557, 0.08492966};
    static constexpr Array hue_means = {6.96195504, -2.89108923, 11.31023104, 0.57595815, 4.04752825, -5.55301231, 9.69434493, -1.34614132};
    static constexpr Array hue_covs = {0.38831633, 0.43726528, 0.33392143, 0.32444862, 1.07510566, 0.19457601, 0.35457794, 0.15187700};
    
    static constexpr Array sat_weights = {0.09713084, 0.13469279, 0.11840812, 0.15963278, 0.10077283, 0.11777569, 0.13832214, 0.13326481};
    static constexpr Array sat_means = {-5.28501930, 4.79250700, 9.46615962, -0.27033550, 2.54706985, 6.65366799, -2.55780595, 11.54815262};
    static constexpr Array sat_covs = {0.51787753, 0.71454204, 0.81614231, 0.80506982, 0.91408268, 0.72683746, 0.93444933, 0.41576665};

    double gmm1d(double x, const Array& w, const Array& m, const Array& v) const {
        const double PI = 3.141592653589793;
        double density = 0.0;
        for (int i = 0; i < N; ++i) {
            double diff = x - m[i];
            density += w[i] * std::exp(-(diff * diff) / (2.0 * v[i])) / std::sqrt(2.0 * PI * v[i]);
        }
        return density;
    }
};

// --- 工具函数 ---

// 辅助函数：应用矩阵变换
Vec3 multiply_matrix(const cgltf_float m[16], Vec3 v) {
    return {
        v.x * m[0] + v.y * m[4] + v.z * m[8] + m[12],
        v.x * m[1] + v.y * m[5] + v.z * m[9] + m[13],
        v.x * m[2] + v.y * m[6] + v.z * m[10] + m[14]
    };
}

HSV rgb_to_hsv(Color c) {
    double r = c.r / 255.0, g = c.g / 255.0, b = c.b / 255.0;
    double max = std::max({r, g, b}), min = std::min({r, g, b});
    double delta = max - min;
    HSV hsv = {0, 0, max};
    if (delta > 0) {
        hsv.s = delta / max;
        if (max == r) hsv.h = 60.0 * fmod((g - b) / delta, 6.0);
        else if (max == g) hsv.h = 60.0 * ((b - r) / delta + 2.0);
        else hsv.h = 60.0 * ((r - g) / delta + 4.0);
        if (hsv.h < 0) hsv.h += 360.0;
    }
    return hsv;
}

// 颜色归一化：将 RGB 的最大值拉伸到 255
Color normalize_to_255(Color c, float& out_ko) {
    float max_val = std::max({(float)c.r, (float)c.g, (float)c.b, 1.0f});
    out_ko = max_val / 255.0f;
    return { (uint8_t)(c.r / max_val * 255), (uint8_t)(c.g / max_val * 255), (uint8_t)(c.b / max_val * 255) };
}

// 1. 新增：射线与三角形求交 (Moller-Trumbore 算法)
bool ray_intersects_triangle(Vec3 ray_o, Vec3 ray_d, Vec3 v0, Vec3 v1, Vec3 v2) {
    Vec3 edge1 = v1 - v0;
    Vec3 edge2 = v2 - v0;
    Vec3 h = ray_d.cross(edge2);
    float a = edge1.dot(h);
    if (a > -1e-6f && a < 1e-6f) return false;
    float f = 1.0f / a;
    Vec3 s = ray_o - v0;
    float u = f * s.dot(h);
    if (u < 0.0f || u > 1.0f) return false;
    Vec3 q = s.cross(edge1);
    float v = f * ray_d.dot(q);
    if (v < 0.0f || u + v > 1.0f) return false;
    float t = f * edge2.dot(q);
    return t > 1e-6f;
}

// ΔP 感知距离计算
double calculate_delta_p(Color c1, Color c2, const Keyness2D& kn) {
    auto hsv1 = rgb_to_hsv(c1);
    auto hsv2 = rgb_to_hsv(c2);
    double dh = std::abs(hsv1.h - hsv2.h);
    if (dh > 180.0) dh = 360.0 - dh;
    double ds = std::abs(hsv1.s - hsv2.s);
    double weight = kn.keyness(hsv1.h, hsv1.s);
    // 使用方程: ΔP = √[ (ΔH × K_h(h))² + (ΔS × K_s(h) × (1 - s)^0.6)² ]
    double term_h = dh * weight;
    double term_s = ds * weight * std::pow(1.0 - hsv1.s, 0.6);
    return std::sqrt(term_h * term_h + term_s * term_s);
}

class GlobalPalette {
public:
    std::vector<Color> colors;
    Keyness2D kn;

    void add_to_library(Color raw_color) {
        float dummy_ko;
        Color base = normalize_to_255(raw_color, dummy_ko);
        for (const auto& exist : colors) {
            if (calculate_delta_p(base, exist, kn) < 0.05) return; // Ji 阈值判定
        }
        if (colors.size() < 1024) colors.push_back(base);
    }

    int find_best_match(Color target_raw) {
        float dummy_ko;
        Color base_target = normalize_to_255(target_raw, dummy_ko);
        int best_id = 0; double min_dp = 1e10;
        for (int i = 0; i < colors.size(); ++i) {
            double dp = calculate_delta_p(base_target, colors[i], kn);
            if (dp < min_dp) { min_dp = dp; best_id = i; }
        }
        return best_id;
    }

    void load(const std::string& path) {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs.is_open()) return;
        colors.resize(1024);
        ifs.seekg(8); // 跳过 CPAL(4) + Count(4)
        ifs.read((char*)colors.data(), 1024 * 3);
    }

    void save(const std::string& path) {
        std::ofstream ofs(path, std::ios::binary);
        ofs.write("CPAL", 4);
        uint32_t count = 1024;
        ofs.write((char*)&count, 4);
        std::vector<Color> output = colors;
        output.resize(1024, {0,0,0});
        ofs.write((char*)output.data(), 1024 * 3);
    }
};

// --- CEM4 256-bit 细胞结构 (32字节对齐) ---
// 严格对齐，禁止编译器填充
#pragma pack(push, 1)
struct VoxelCEM4 {
    uint32_t word0; // [SDF: 20-bit][ColorID: 12-bit]
    uint32_t word1; // [Normal: 20-bit][Ko: 8-bit][Flags: 4-bit]
    uint32_t word2; // [Aniso_Stretch: 3x10-bit][Reserved: 2-bit]
    uint32_t word3; // [Tangent_Oct: 14-bit][Emissive: 8-bit][AO: 8-bit][Res: 2-bit]
    uint32_t child_ptr; // 32-bit SVO 节点偏移或 Brick 索引
    uint32_t tetra_info; // [TetraID: 16-bit][Animation_Flags: 16-bit]
    uint32_t res_word6; // 预留 (风力影响强度)
    uint32_t res_word7; // 预留 (厚度感/流体系数)

    VoxelCEM4() { memset(this, 0, sizeof(VoxelCEM4)); }
};
#pragma pack(pop)

// --- 材质来源追踪结构 ---
struct TriangleSource {
    int mesh_idx;
    int prim_idx;
    std::string node_name;
};

// --- MeshData 结构 ---
struct MeshData {
    std::vector<Vec3> vertices;
    std::vector<Vec3> normals; // 新增：存法线
    std::vector<Color> colors;
    std::vector<Vec2> uvs; // 新增：存储每个顶点的 UV
    std::vector<uint32_t> indices;
    std::vector<TriangleSource> tri_sources; // 增加：三角形来源追踪
    
    // 增加对贴图的引用
    uint8_t* texture_data = nullptr;
    int tex_w, tex_h;
    
    Vec3 min_b, max_b;
};

// --- 1. 辅助结构：三角形 AABB ---
struct TriBounds {
    Vec3 min_b, max_b;
    int index;
};

// --- 2. 升级版点云收集器：密度感知采样 ---
// 为每个体素寻找半径范围内的几何点
void collect_neighborhood_points(
    const Vec3& voxel_p, 
    float radius, 
    const MeshData& mesh, 
    const std::vector<TriBounds>& tri_indices, 
    std::vector<Vec3>& out_points,      // 改为引用传递，避免拷贝
    std::mt19937& rng                 // 传入随机数生成器
) {
    out_points.clear();
    float r_sq = radius * radius;
    std::uniform_real_distribution<float> distrib(0.0f, 1.0f);

    for (const auto& tri_box : tri_indices) {
        // 1. 快速 AABB 排除
        if (voxel_p.x + radius < tri_box.min_b.x || voxel_p.x - radius > tri_box.max_b.x ||
            voxel_p.y + radius < tri_box.min_b.y || voxel_p.y - radius > tri_box.max_b.y ||
            voxel_p.z + radius < tri_box.min_b.z || voxel_p.z - radius > tri_box.max_b.z) {
            continue;
        }

        // 2. 拿到三角形顶点
        int idx = tri_box.index;
        Vec3 v0 = mesh.vertices[mesh.indices[idx]];
        Vec3 v1 = mesh.vertices[mesh.indices[idx+1]];
        Vec3 v2 = mesh.vertices[mesh.indices[idx+2]];

        // --- 核心修复：基于面积的自适应采样 ---
        float area = (v1 - v0).cross(v2 - v0).length() * 0.5f;
        // 根据三角形在本邻域内的可见面积，决定采样点数，确保样本密度稳定
        int num_samples = std::clamp((int)(area / (radius * radius) * 50.0f), 2, 8);
        
        for (int i = 0; i < num_samples; ++i) {
            // 使用高质量随机数生成重心坐标
            float r1 = distrib(rng);
            float r2 = distrib(rng);
            if (r1 + r2 > 1.0f) {
                r1 = 1.0f - r1;
                r2 = 1.0f - r2;
            }
            Vec3 sample_p = v0 * (1.0f - r1 - r2) + v1 * r1 + v2 * r2;
            
            // 只收集在严格半径内的点
            if ((sample_p - voxel_p).length_sq() <= r_sq) {
                out_points.push_back(sample_p);
            }
        }
    }
}

// --- 3. 保存 CEM4 文件 ---
void save_cem4(const std::string& path, int res, Vec3 min_b, Vec3 max_b, const std::vector<VoxelCEM4>& voxels) {
    std::ofstream ofs(path, std::ios::binary);
    ofs.write("CEM4", 4);
    ofs.write((char*)&res, 4);
    ofs.write((char*)&min_b, 12);
    ofs.write((char*)&max_b, 12);
    ofs.write((char*)voxels.data(), voxels.size() * 32);
}

// --- 高精度八面体编码 (支持动态位宽) ---
uint32_t encode_oct(Vec3 n, int bits_per_axis) {
    if (n.length_sq() < 1e-6f) return 0;
    n = n * (1.0f / n.length());
    float l1 = std::abs(n.x) + std::abs(n.y) + std::abs(n.z);
    float x = n.x / l1;
    float y = n.y / l1;
    if (n.z < 0.0f) {
        float tx = (1.0f - std::abs(y)) * (x >= 0.0f ? 1.0f : -1.0f);
        float ty = (1.0f - std::abs(x)) * (y >= 0.0f ? 1.0f : -1.0f);
        x = tx; y = ty;
    }
    uint32_t mask = (1 << bits_per_axis) - 1;
    uint32_t ux = (uint32_t)std::clamp((x * 0.5f + 0.5f) * mask, 0.0f, (float)mask);
    uint32_t uy = (uint32_t)std::clamp((y * 0.5f + 0.5f) * mask, 0.0f, (float)mask);
    return (ux << bits_per_axis) | uy;
}

// Jacobi 旋转法：求解 3x3 对称矩阵特征值
void solve_eigen_system(float cov[3][3], float eigenvalues[3], Vec3 eigenvectors[3]) {
    float v[3][3] = {{1,0,0}, {0,1,0}, {0,0,1}};
    for (int iter = 0; iter < 15; ++iter) {
        int p = 0, q = 1;
        float max_v = std::abs(cov[0][1]);
        if (std::abs(cov[0][2]) > max_v) { p = 0; q = 2; max_v = std::abs(cov[0][2]); }
        if (std::abs(cov[1][2]) > max_v) { p = 1; q = 2; max_v = std::abs(cov[1][2]); }
        if (max_v < 1e-7f) break;

        float theta = 0.5f * std::atan2(2.0f * cov[p][q], cov[q][q] - cov[p][p]);
        float s = std::sin(theta), c = std::cos(theta);
        
        float cpp = c*c*cov[p][p] - 2*s*c*cov[p][q] + s*s*cov[q][q];
        float cqq = s*s*cov[p][p] + 2*s*c*cov[p][q] + c*c*cov[q][q];
        cov[p][p] = cpp; cov[q][q] = cqq; cov[p][q] = cov[q][p] = 0.0f;

        for (int i = 0; i < 3; ++i) {
            float vip = c * v[i][p] - s * v[i][q];
            float viq = s * v[i][p] + c * v[i][q];
            v[i][p] = vip; v[i][q] = viq;
        }
    }
    for (int i = 0; i < 3; ++i) eigenvalues[i] = cov[i][i];
    for (int i = 0; i < 3; ++i) eigenvectors[i] = {v[0][i], v[1][i], v[2][i]};
    
    // 排序 e1 > e2 > e3
    if (eigenvalues[0] < eigenvalues[1]) { std::swap(eigenvalues[0], eigenvalues[1]); std::swap(eigenvectors[0], eigenvectors[1]); }
    if (eigenvalues[0] < eigenvalues[2]) { std::swap(eigenvalues[0], eigenvalues[2]); std::swap(eigenvectors[0], eigenvectors[2]); }
    if (eigenvalues[1] < eigenvalues[2]) { std::swap(eigenvalues[1], eigenvalues[2]); std::swap(eigenvectors[1], eigenvectors[2]); }
}

// --- 核心烘焙函数：形状感知 (Baker 2.0) ---

void bake_cem4_voxel(VoxelCEM4& voxel, float dist, int color_id, Vec3 smooth_n, float ko, 
                    const std::vector<Vec3>& neighborhood_points, float world_size) {
    
    // 1. 打包 Word0: 高精度 SDF + 颜色
    float normalized_d = std::clamp(dist / world_size, -1.0f, 1.0f);
    int32_t d_int = (int32_t)(normalized_d * 524287.0f);
    voxel.word0 = ((uint32_t)d_int & 0xFFFFF) << 12 | (color_id & 0xFFF);

    // 2. 打包 Word1: 注入法线 + Ko
    voxel.word1 = (encode_oct(smooth_n, 10) << 12) | ((uint32_t)(ko * 255.0f) << 4);

    // 3. PCA 各向异性分析（解决细杆、薄片的核心）
    if (neighborhood_points.size() > 10) { // 提高点云下限，过滤掉稀疏区域
        Vec3 centroid = {0,0,0};
        for (auto& p : neighborhood_points) centroid = centroid + p;
        centroid = centroid * (1.0f / neighborhood_points.size());

        float cov[3][3] = {0};
        for (auto& p : neighborhood_points) {
            Vec3 d = p - centroid;
            cov[0][0]+=d.x*d.x; cov[0][1]+=d.x*d.y; cov[0][2]+=d.x*d.z;
            cov[1][1]+=d.y*d.y; cov[1][2]+=d.y*d.z; cov[2][2]+=d.z*d.z;
        }
        cov[1][0]=cov[0][1]; cov[2][0]=cov[0][2]; cov[2][1]=cov[1][2];

        float evals[3]; Vec3 evecs[3];
        solve_eigen_system(cov, evals, evecs);

        // --- 核心修复：主轴方向对齐 (解决彩虹树) ---
        // 我们用插值出的平滑法线作为参考方向，确保局部切线的一致性
        if (evecs[0].dot(smooth_n) < 0) {
            evecs[0] = evecs[0] * -1.0f; // 翻转主轴
        }
        // 对于次轴和副轴，我们用叉乘保证右手坐标系，防止镜像
        evecs[2] = evecs[0].cross(evecs[1]);
        if (evecs[2].length_sq() < 1e-6f) { // 防止共线
             evecs[2] = evecs[0].cross({1,0,0});
             if(evecs[2].length_sq() < 1e-6f) evecs[2] = evecs[0].cross({0,1,0});
        }
        evecs[2] = evecs[2] * (1.0f / evecs[2].length());
        evecs[1] = evecs[2].cross(evecs[0]);

        // Word2: 各向异性拉伸 (sx, sy, sz)
        float sum_e = evals[0] + evals[1] + evals[2] + 1e-8f;
        uint32_t s0 = (uint32_t)(std::max(0.0f, evals[0]) / sum_e * 1023.0f);
        uint32_t s1 = (uint32_t)(std::max(0.0f, evals[1]) / sum_e * 1023.0f);
        uint32_t s2 = (uint32_t)(std::max(0.0f, evals[2]) / sum_e * 1023.0f);
        voxel.word2 = (s0 << 22) | (s1 << 12) | (s2 << 2);

        // Word3: 切空间主轴 (Tangent)
        uint32_t tang_oct = encode_oct(evecs[0], 7); // 14-bit
        voxel.word3 = (tang_oct << 18);
    }
}

// --- 几何重心计算 (为了拿到平滑法线) ---
Vec3 get_closest_point_barycentric(Vec3 p, Vec3 a, Vec3 b, Vec3 c, float& u, float& v, float& w) {
    Vec3 v0 = b - a, v1 = c - a, v2 = p - a;
    float d00 = v0.dot(v0); float d01 = v0.dot(v1); float d11 = v1.dot(v1);
    float d20 = v2.dot(v0); float d21 = v2.dot(v1);
    float denom = d00 * d11 - d01 * d01;
    v = (d11 * d20 - d01 * d21) / denom;
    w = (d00 * d21 - d01 * d20) / denom;
    u = 1.0f - v - w;
    u = std::clamp(u, 0.0f, 1.0f); v = std::clamp(v, 0.0f, 1.0f); w = std::clamp(w, 0.0f, 1.0f);
    return a * u + b * v + c * w;
}

// --- 其他几何与导出函数 (保持逻辑一致) ---

float point_to_tri_dist_sq(Vec3 p, Vec3 a, Vec3 b, Vec3 c, Vec3& out_normal) {
    Vec3 ab = b - a; Vec3 bc = c - b; Vec3 ca = a - c;
    Vec3 ap = p - a; Vec3 bp = p - b; Vec3 cp = p - c;
    Vec3 n = ab.cross(ca);
    out_normal = n;
    auto project_on_seg = [](Vec3 p, Vec3 a, Vec3 b) {
        Vec3 ab = b - a;
        float t = std::clamp((p - a).dot(ab) / ab.length_sq(), 0.0f, 1.0f);
        return (p - (a + ab * t)).length_sq();
    };
    bool inside = ab.cross(n).dot(ap) >= 0.0f && bc.cross(n).dot(bp) >= 0.0f && ca.cross(n).dot(cp) >= 0.0f;
    if (inside) return (n.dot(ap) * n.dot(ap)) / n.length_sq();
    return std::min({project_on_seg(p, a, b), project_on_seg(p, b, c), project_on_seg(p, c, a)});
}

// 递归遍历节点
void process_node(cgltf_node* node, const cgltf_float parent_matrix[16], MeshData& mesh, int& img_comp) {
    cgltf_float global_matrix[16];
    cgltf_node_transform_world(node, global_matrix);

    if (node->mesh) {
        printf("Node [%s] detected. World Offset Y: %f\n", node->name ? node->name : "unnamed", global_matrix[13]);
        
        for (int i = 0; i < node->mesh->primitives_count; ++i) {
            cgltf_primitive* prim = &node->mesh->primitives[i];
            
            // 读取并保留贴图数据
            if (prim->material && prim->material->has_pbr_metallic_roughness && prim->material->pbr_metallic_roughness.base_color_texture.texture) {
                auto* view = prim->material->pbr_metallic_roughness.base_color_texture.texture->image->buffer_view;
                mesh.texture_data = stbi_load_from_memory(
                    (uint8_t*)view->buffer->data + view->offset,
                    view->size, &mesh.tex_w, &mesh.tex_h, &img_comp, 3
                );
                printf("  - Texture Loaded: %dx%d\n", mesh.tex_w, mesh.tex_h);
            }

            size_t vertex_offset = mesh.vertices.size();
            cgltf_accessor *pos_acc = nullptr, *uv_acc = nullptr, *col_acc = nullptr, *norm_acc = nullptr;
            for (int j = 0; j < prim->attributes_count; ++j) {
                if (prim->attributes[j].type == cgltf_attribute_type_position) pos_acc = prim->attributes[j].data;
                if (prim->attributes[j].type == cgltf_attribute_type_texcoord) uv_acc = prim->attributes[j].data;
                if (prim->attributes[j].type == cgltf_attribute_type_color) col_acc = prim->attributes[j].data;
                if (prim->attributes[j].type == cgltf_attribute_type_normal) norm_acc = prim->attributes[j].data;
            }
            for (int k = 0; k < pos_acc->count; ++k) {
                float v[3]; cgltf_accessor_read_float(pos_acc, k, v, 3);
                // 应用变换矩阵
                Vec3 p_local = {v[0], v[1], v[2]};
                Vec3 p_world = multiply_matrix(global_matrix, p_local);
                mesh.vertices.push_back(p_world);
                mesh.min_b.x = std::min(mesh.min_b.x, p_world.x); mesh.max_b.x = std::max(mesh.max_b.x, p_world.x);
                mesh.min_b.y = std::min(mesh.min_b.y, p_world.y); mesh.max_b.y = std::max(mesh.max_b.y, p_world.y);
                mesh.min_b.z = std::min(mesh.min_b.z, p_world.z); mesh.max_b.z = std::max(mesh.max_b.z, p_world.z);
                
                // 读取 UV
                if (uv_acc) {
                    float uv[2];
                    cgltf_accessor_read_float(uv_acc, k, uv, 2);
                    mesh.uvs.push_back({uv[0], uv[1]});
                } else {
                    mesh.uvs.push_back({0, 0});
                }
                
                Color c = {255, 255, 255};
                if (col_acc) { float fc[4]; cgltf_accessor_read_float(col_acc, k, fc, 4); c = {(uint8_t)(fc[0]*255), (uint8_t)(fc[1]*255), (uint8_t)(fc[2]*255)}; }
                else if (mesh.texture_data && !mesh.uvs.empty()) { 
                    float uv[2]; cgltf_accessor_read_float(uv_acc, k, uv, 2);
                    int x = (int)(uv[0] * mesh.tex_w) % mesh.tex_w, y = (int)(uv[1] * mesh.tex_h) % mesh.tex_h;
                    int idx = (y * mesh.tex_w + (x < 0 ? x + mesh.tex_w : x)) * 3;
                    c = {mesh.texture_data[idx], mesh.texture_data[idx+1], mesh.texture_data[idx+2]};
                }
                mesh.colors.push_back(c);
                if (norm_acc) {
                    float fn[3]; cgltf_accessor_read_float(norm_acc, k, fn, 3);
                    mesh.normals.push_back({fn[0], fn[1], fn[2]});
                } else {
                    mesh.normals.push_back({0, 1, 0}); // 兜底
                }
            }
            if (prim->indices) {
                for (int k = 0; k < prim->indices->count; k += 3) {
                    mesh.indices.push_back(vertex_offset + cgltf_accessor_read_index(prim->indices, k));
                    mesh.indices.push_back(vertex_offset + cgltf_accessor_read_index(prim->indices, k+1));
                    mesh.indices.push_back(vertex_offset + cgltf_accessor_read_index(prim->indices, k+2));
                    mesh.tri_sources.push_back({(int)0, i, node->name ? node->name : "unnamed"});
                }
            }
        }
    }

    for (int i = 0; i < node->children_count; ++i) {
        process_node(node->children[i], global_matrix, mesh, img_comp);
    }
}

MeshData load_glb(const std::string& path) {
    cgltf_options options = {};
    cgltf_data* data = nullptr;
    if (cgltf_parse_file(&options, path.c_str(), &data) != cgltf_result_success) exit(-1);
    cgltf_load_buffers(&options, data, path.c_str());
    MeshData mesh; mesh.min_b = {1e10, 1e10, 1e10}; mesh.max_b = {-1e10, -1e10, -1e10};
    int img_comp = 0;
    
    printf("\n--- GLB Node Topology Debug ---\n");
    
    // 初始矩阵（单位矩阵）
    cgltf_float identity_matrix[16] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    
    // 遍历所有根节点
    for (size_t n_idx = 0; n_idx < data->nodes_count; ++n_idx) {
        process_node(&data->nodes[n_idx], identity_matrix, mesh, img_comp);
    }
    
    printf("Global AABB: Min(%f, %f, %f) Max(%f, %f, %f)\n", 
           mesh.min_b.x, mesh.min_b.y, mesh.min_b.z, 
           mesh.max_b.x, mesh.max_b.y, mesh.max_b.z);
    
    cgltf_free(data); return mesh;
}

#include <filesystem>

void generate_global_palette(const std::string& glb_dir, const std::string& output_cpal) {
    namespace fs = std::filesystem;
    GlobalPalette pal;
    pal.add_to_library({255, 255, 255}); // 强制白
    pal.add_to_library({128, 128, 128}); // 强制灰
    
    // 遍历文件夹中的所有 GLB 文件
    for (const auto& entry : fs::directory_iterator(glb_dir)) {
        if (entry.path().extension() == ".glb") {
            std::string file = entry.path().filename().string();
            std::cout << "Processing: " << file << std::endl;
            MeshData mesh = load_glb(glb_dir + "\\" + file);
            for (const auto& c : mesh.colors) pal.add_to_library(c);
        }
    }
    
    pal.save(output_cpal);
    std::cout << "Palette Generated. Unique Hue-Bases: " << pal.colors.size() << std::endl;
}

// --- 升级版 Cook 函数 (CEM4) ---
void cook_cem_v4(const std::string& input_glb, const std::string& output_cem, GlobalPalette& pal) {
    MeshData mesh = load_glb(input_glb);
    int res = 64;
    std::vector<VoxelCEM4> voxels(res * res * res);
    Vec3 size = mesh.max_b - mesh.min_b;
    float world_size = std::max({size.x, size.y, size.z});
    float voxel_unit = world_size / (float)res;

    std::cout << "Step 2: Cooking CEM4 (256-bit) with PCA Shape Analysis..." << std::endl;

    // 预处理三角形 AABB 索引，加速搜索
    std::vector<TriBounds> tri_bounds;
    for (size_t i = 0; i < mesh.indices.size(); i += 3) {
        Vec3 v0 = mesh.vertices[mesh.indices[i]];
        Vec3 v1 = mesh.vertices[mesh.indices[i+1]];
        Vec3 v2 = mesh.vertices[mesh.indices[i+2]];
        Vec3 mi = { std::min({v0.x, v1.x, v2.x}), std::min({v0.y, v1.y, v2.y}), std::min({v0.z, v1.z, v2.z}) };
        Vec3 ma = { std::max({v0.x, v1.x, v2.x}), std::max({v0.y, v1.y, v2.y}), std::max({v0.z, v1.z, v2.z}) };
        tri_bounds.push_back({mi, ma, (int)i});
    }

    // 初始化高质量随机数引擎
    std::mt19937 rng(std::random_device{}());
    std::vector<Vec3> neighborhood; // 循环外复用内存，提升性能

    for (int z = 0; z < res; ++z) {
        for (int y = 0; y < res; ++y) {
            for (int x = 0; x < res; ++x) {
                Vec3 p = {
                    mesh.min_b.x + (x / (float)(res-1)) * size.x,
                    mesh.min_b.y + (y / (float)(res-1)) * size.y,
                    mesh.min_b.z + (z / (float)(res-1)) * size.z
                };

                // 1. 寻找最近三角形并计算基础 SDF
                float min_d_sq = 1e10f; 
                int closest_tri = 0;
                float cu, cv, cw;

                for (size_t i = 0; i < tri_bounds.size(); ++i) {
                    int tri_idx = tri_bounds[i].index;
                    Vec3 dummy_n;
                    float d_sq = point_to_tri_dist_sq(p, mesh.vertices[mesh.indices[tri_idx]], 
                                                       mesh.vertices[mesh.indices[tri_idx+1]], 
                                                       mesh.vertices[mesh.indices[tri_idx+2]], dummy_n);
                    if (d_sq < min_d_sq) {
                        min_d_sq = d_sq; 
                        closest_tri = tri_idx;
                        get_closest_point_barycentric(p, mesh.vertices[mesh.indices[tri_idx]], 
                                                       mesh.vertices[mesh.indices[tri_idx+1]], 
                                                       mesh.vertices[mesh.indices[tri_idx+2]], cu, cv, cw);
                    }
                }

                // 2. 射线投票判定符号
                int hit_count = 0;
                Vec3 ray_dir = {1.0f, 0.432f, 0.123f}; 
                ray_dir = ray_dir * (1.0f / ray_dir.length());
                for (size_t i = 0; i < mesh.indices.size(); i += 3) {
                    if (ray_intersects_triangle(p, ray_dir, mesh.vertices[mesh.indices[i]], 
                                               mesh.vertices[mesh.indices[i+1]], 
                                               mesh.vertices[mesh.indices[i+2]])) hit_count++;
                }
                float dist = std::sqrt(min_d_sq);
                if (hit_count % 2 != 0) dist = -dist;

                // 3. 调用强化版点云收集器
                collect_neighborhood_points(p, voxel_unit * 1.5f, mesh, tri_bounds, neighborhood, rng);

                // 4. 获取法线与颜色 (保持原有逻辑)
                Vec3 n1 = mesh.normals[mesh.indices[closest_tri]];
                Vec3 n2 = mesh.normals[mesh.indices[closest_tri+1]];
                Vec3 n3 = mesh.normals[mesh.indices[closest_tri+2]];
                Vec3 smooth_n = (n1 * cu + n2 * cv + n3 * cw) * (1.0f / (cu + cv + cw + 1e-8f));
                smooth_n = smooth_n * (1.0f / smooth_n.length());
                
                Color raw_c = mesh.colors[mesh.indices[closest_tri]];
                if (mesh.texture_data && !mesh.uvs.empty()) {
                    Vec2 uv1 = mesh.uvs[mesh.indices[closest_tri]];
                    Vec2 uv2 = mesh.uvs[mesh.indices[closest_tri + 1]];
                    Vec2 uv3 = mesh.uvs[mesh.indices[closest_tri + 2]];

                    float u = uv1.u * cu + uv2.u * cv + uv3.u * cw;
                    float v = uv1.v * cu + uv2.v * cv + uv3.v * cw;

                    int tx = (int)(u * mesh.tex_w) % mesh.tex_w;
                    int ty = (int)(v * mesh.tex_h) % mesh.tex_h;
                    if (tx < 0) tx += mesh.tex_w;
                    if (ty < 0) ty += mesh.tex_h;

                    int pixel_idx = (ty * mesh.tex_w + tx) * 3;
                    raw_c.r = mesh.texture_data[pixel_idx];
                    raw_c.g = mesh.texture_data[pixel_idx + 1];
                    raw_c.b = mesh.texture_data[pixel_idx + 2];
                }

                float ko;
                Color norm_c = normalize_to_255(raw_c, ko);

                // 5. 调用带主轴对齐的 Baker
                VoxelCEM4 voxel;
                bake_cem4_voxel(voxel, dist, pal.find_best_match(raw_c), smooth_n, ko, neighborhood, world_size);
                
                voxels[x + y*res + z*res*res] = voxel;
            }
        }
    }

    // 7. 计算邻域安全距离 (Regional Safety Sphere)
    std::cout << "Step 3: Calculating Neighborhood Safety Distances (5x5x5)..." << std::endl;
    for (int z = 0; z < res; ++z) {
        for (int y = 0; y < res; ++y) {
            for (int x = 0; x < res; ++x) {
                float min_abs_dist = 1.0f;
                // 扩大到 5x5x5 邻域 (探测范围翻倍)
                for (int dz = -2; dz <= 2; ++dz) {
                    for (int dy = -2; dy <= 2; ++dy) {
                        for (int dx = -2; dx <= 2; ++dx) {
                            int nx = std::clamp(x + dx, 0, res - 1);
                            int ny = std::clamp(y + dy, 0, res - 1);
                            int nz = std::clamp(z + dz, 0, res - 1);

                            // 从 word0 中提取 SDF 值
                            uint32_t word0 = voxels[nx + ny * res + nz * res * res].word0;
                            uint32_t d_bits = (word0 >> 12u) & 0xFFFFFu;
                            int32_t d_i32 = static_cast<int32_t>(d_bits);
                            if (d_bits >= 524288u) { d_i32 -= 1048576; }
                            float d = static_cast<float>(d_i32) / 524287.0f;
                            min_abs_dist = std::min(min_abs_dist, std::abs(d));
                        }
                    }
                }
                // 溶解进保留字段：映射到 0-255 的字节
                voxels[x + y * res + z * res * res].res_word6 = static_cast<uint32_t>(min_abs_dist * 255.0f);
            }
        }
    }

    // 6. 保存 CEM4 文件
    save_cem4(output_cem, res, mesh.min_b, mesh.max_b, voxels);
    
    // 释放贴图数据
    if (mesh.texture_data) stbi_image_free(mesh.texture_data);
}

void cook_all(const std::string& glb_dir, const std::string& palette_path, const std::string& output_dir) {
    namespace fs = std::filesystem;
    
    // 加载调色板
    GlobalPalette pal;
    pal.load(palette_path);
    
    // 确保输出目录存在
    fs::create_directories(output_dir);
    
    // 遍历文件夹中的所有 GLB 文件
    for (const auto& entry : fs::directory_iterator(glb_dir)) {
        if (entry.path().extension() == ".glb") {
            std::string file = entry.path().filename().string();
            std::string input_glb = entry.path().string();
            std::string output_cem = output_dir + "\\" + file.substr(0, file.size() - 4) + ".cem";
            
            std::cout << "Processing: " << file << " -> " << output_cem << std::endl;
            cook_cem_v4(input_glb, output_cem, pal);
        }
    }
    
    std::cout << "Cook-all completed!" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 4) return -1;
    std::string mode = argv[1];
    if (mode == "generate-palette") generate_global_palette(argv[2], argv[3]);
    else if (mode == "cook-cem4") { GlobalPalette pal; pal.load(argv[3]); cook_cem_v4(argv[2], argv[4], pal); }
    else if (mode == "cook-all") {
        if (argc < 5) return -1;
        cook_all(argv[2], argv[3], argv[4]);
    }
    return 0;
}