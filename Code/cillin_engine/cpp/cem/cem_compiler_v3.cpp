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

// --- 其他几何与导出函数 (保持逻辑一致) ---

struct MeshData {
    std::vector<Vec3> vertices;
    std::vector<Color> colors;
    std::vector<uint32_t> indices;
    Vec3 min_b, max_b;
};

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

uint32_t pack_voxel_v2(float dist, int color_id, float modifier, float world_size) {
    float normalized_d = std::clamp(dist / world_size, -1.0f, 1.0f);
    int32_t d_int = (int32_t)(normalized_d * 2047.0f);
    uint32_t d_bits = (uint32_t)d_int & 0xFFF;
    uint32_t id_bits = (uint32_t)color_id & 0x3FF;
    uint32_t mod_bits = (uint32_t)(std::clamp(modifier, 0.0f, 1.0f) * 255.0f);
    return (d_bits << 20) | (id_bits << 10) | (mod_bits << 2);
}

MeshData load_glb(const std::string& path) {
    cgltf_options options = {};
    cgltf_data* data = nullptr;
    if (cgltf_parse_file(&options, path.c_str(), &data) != cgltf_result_success) exit(-1);
    cgltf_load_buffers(&options, data, path.c_str());
    MeshData mesh; mesh.min_b = {1e10, 1e10, 1e10}; mesh.max_b = {-1e10, -1e10, -1e10};
    for (size_t m_idx = 0; m_idx < data->meshes_count; ++m_idx) {
        for (int i = 0; i < data->meshes[m_idx].primitives_count; ++i) {
            cgltf_primitive* prim = &data->meshes[m_idx].primitives[i];
            uint8_t* img_data = nullptr; int img_w, img_h, img_comp;
            if (prim->material && prim->material->has_pbr_metallic_roughness && prim->material->pbr_metallic_roughness.base_color_texture.texture) {
                auto* view = prim->material->pbr_metallic_roughness.base_color_texture.texture->image->buffer_view;
                img_data = stbi_load_from_memory((uint8_t*)view->buffer->data + view->offset, view->size, &img_w, &img_h, &img_comp, 3);
            }
            size_t vertex_offset = mesh.vertices.size();
            cgltf_accessor *pos_acc = nullptr, *uv_acc = nullptr, *col_acc = nullptr;
            for (int j = 0; j < prim->attributes_count; ++j) {
                if (prim->attributes[j].type == cgltf_attribute_type_position) pos_acc = prim->attributes[j].data;
                if (prim->attributes[j].type == cgltf_attribute_type_texcoord) uv_acc = prim->attributes[j].data;
                if (prim->attributes[j].type == cgltf_attribute_type_color) col_acc = prim->attributes[j].data;
            }
            for (int k = 0; k < pos_acc->count; ++k) {
                float v[3]; cgltf_accessor_read_float(pos_acc, k, v, 3);
                Vec3 p = {v[0], v[1], v[2]}; mesh.vertices.push_back(p);
                mesh.min_b.x = std::min(mesh.min_b.x, p.x); mesh.max_b.x = std::max(mesh.max_b.x, p.x);
                mesh.min_b.y = std::min(mesh.min_b.y, p.y); mesh.max_b.y = std::max(mesh.max_b.y, p.y);
                mesh.min_b.z = std::min(mesh.min_b.z, p.z); mesh.max_b.z = std::max(mesh.max_b.z, p.z);
                Color c = {255, 255, 255};
                if (col_acc) { float fc[4]; cgltf_accessor_read_float(col_acc, k, fc, 4); c = {(uint8_t)(fc[0]*255), (uint8_t)(fc[1]*255), (uint8_t)(fc[2]*255)}; }
                else if (img_data && uv_acc) { 
                    float uv[2]; cgltf_accessor_read_float(uv_acc, k, uv, 2);
                    int x = (int)(uv[0] * img_w) % img_w, y = (int)(uv[1] * img_h) % img_h;
                    int idx = (y * img_w + (x < 0 ? x + img_w : x)) * 3;
                    c = {img_data[idx], img_data[idx+1], img_data[idx+2]};
                }
                mesh.colors.push_back(c);
            }
            if (prim->indices) for (int k = 0; k < prim->indices->count; ++k) mesh.indices.push_back(vertex_offset + cgltf_accessor_read_index(prim->indices, k));
            if (img_data) stbi_image_free(img_data);
        }
    }
    cgltf_free(data); return mesh;
}

void generate_global_palette(const std::string& glb_dir, const std::string& output_cpal) {
    std::vector<std::string> glb_files = { "ComeCube.glb", "tree.glb" };
    GlobalPalette pal;
    pal.add_to_library({255, 255, 255}); // 强制白
    pal.add_to_library({128, 128, 128}); // 强制灰
    for (const auto& file : glb_files) {
        MeshData mesh = load_glb(glb_dir + "\\" + file);
        for (const auto& c : mesh.colors) pal.add_to_library(c);
    }
    pal.save(output_cpal);
    std::cout << "Palette Generated. Unique Hue-Bases: " << pal.colors.size() << std::endl;
}

void cook_cem_v2(const std::string& input_glb, const std::string& output_cem, GlobalPalette& pal) {
    MeshData mesh = load_glb(input_glb);
    int res = 64; std::vector<uint32_t> voxels(res * res * res);
    Vec3 size = mesh.max_b - mesh.min_b; float world_size = std::max({size.x, size.y, size.z});
    for (int z = 0; z < res; ++z) {
        for (int y = 0; y < res; ++y) {
            for (int x = 0; x < res; ++x) {
                Vec3 p = { mesh.min_b.x + (x / (float)(res-1)) * size.x, mesh.min_b.y + (y / (float)(res-1)) * size.y, mesh.min_b.z + (z / (float)(res-1)) * size.z };
                float min_d_sq = 1e10f; int closest_tri = 0; Vec3 closest_n;
                for (size_t i = 0; i < mesh.indices.size(); i += 3) {
                    Vec3 a = mesh.vertices[mesh.indices[i]], b = mesh.vertices[mesh.indices[i+1]], c = mesh.vertices[mesh.indices[i+2]], tri_n;
                    float d_sq = point_to_tri_dist_sq(p, a, b, c, tri_n);
                    if (d_sq < min_d_sq) { min_d_sq = d_sq; closest_tri = i; closest_n = tri_n; }
                }
                float dist = std::sqrt(min_d_sq);
                if ((p - mesh.vertices[mesh.indices[closest_tri]]).dot(closest_n) < 0) dist = -dist;
                
                // 查找三角形内最近顶点的颜色，实现棋盘格
                uint32_t best_v = mesh.indices[closest_tri]; float best_vd = 1e10f;
                for(int k=0; k<3; k++) {
                    float d = (p - mesh.vertices[mesh.indices[closest_tri+k]]).length_sq();
                    if(d < best_vd) { best_vd = d; best_v = mesh.indices[closest_tri+k]; }
                }
                Color raw_c = mesh.colors[best_v];
                float ko; normalize_to_255(raw_c, ko);
                voxels[x + y*res + z*res*res] = pack_voxel_v2(dist, pal.find_best_match(raw_c), ko, world_size);
            }
        }
    }
    std::ofstream ofs(output_cem, std::ios::binary);
    ofs.write("CEM2", 4); ofs.write((char*)&res, 4);
    ofs.write((char*)&mesh.min_b, 12); ofs.write((char*)&mesh.max_b, 12);
    ofs.write((char*)voxels.data(), voxels.size() * 4);
}

int main(int argc, char** argv) {
    if (argc < 4) return -1;
    std::string mode = argv[1];
    if (mode == "generate-palette") generate_global_palette(argv[2], argv[3]);
    else if (mode == "cook") { GlobalPalette pal; pal.load(argv[3]); cook_cem_v2(argv[2], argv[4], pal); }
    return 0;
}