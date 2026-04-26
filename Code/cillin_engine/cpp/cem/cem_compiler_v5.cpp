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

// --- 八面体法线编码 (Octahedron Encoding) ---
// 将 3D 向量完美压入 10 bits，保持极高精度
uint32_t encode_normal_10bit(Vec3 n) {
    float l1 = std::abs(n.x) + std::abs(n.y) + std::abs(n.z);
    float x = n.x / l1;
    float y = n.y / l1;
    if (n.z < 0.0f) {
        float tx = (1.0f - std::abs(y)) * (x >= 0.0f ? 1.0f : -1.0f);
        float ty = (1.0f - std::abs(x)) * (y >= 0.0f ? 1.0f : -1.0f);
        x = tx; y = ty;
    }
    uint32_t ux = (uint32_t)std::clamp((x * 0.5f + 0.5f) * 31.0f, 0.0f, 31.0f);
    uint32_t uy = (uint32_t)std::clamp((y * 0.5f + 0.5f) * 31.0f, 0.0f, 31.0f);
    return (ux << 5) | uy;
}

// --- 64-bit 全息体素结构 ---
struct Voxel64 {
    uint32_t r; // [SDF 20-bit][ColorID 12-bit]
    uint32_t g; // [Normal 10-bit][Ko 8-bit][Reserved 14-bit]
};

Voxel64 pack_voxel_64bit(float dist, int color_id, Vec3 normal, float ko, float world_size) {
    // 1. R 通道：高精度距离 + 颜色
    float normalized_d = std::clamp(dist / world_size, -1.0f, 1.0f);
    int32_t d_int = (int32_t)(normalized_d * 524287.0f); // 20-bit 范围
    uint32_t d_bits = (uint32_t)d_int & 0xFFFFF;
    uint32_t id_bits = (uint32_t)color_id & 0xFFF;
    uint32_t r = (d_bits << 12) | id_bits;

    // 2. G 通道：法线 + Ko (还给你了！)
    uint32_t norm_bits = encode_normal_10bit(normal) & 0x3FF;
    uint32_t ko_bits = (uint32_t)(std::clamp(ko, 0.0f, 1.0f) * 255.0f) & 0xFF;
    uint32_t g = (norm_bits << 22) | (ko_bits << 14); 

    return {r, g};
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

// 材质来源追踪结构
struct TriangleSource {
    int mesh_idx;
    int prim_idx;
    std::string node_name;
};

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

void cook_cem_v2(const std::string& input_glb, const std::string& output_cem, GlobalPalette& pal) {
    MeshData mesh = load_glb(input_glb);
    int res = 64;
    std::vector<uint32_t> voxels(res * res * res);
    Vec3 size = mesh.max_b - mesh.min_b;
    float world_size = std::max({size.x, size.y, size.z});

    std::cout << "Cooking with Ray-Casting Sign Detection..." << std::endl;

    for (int z = 0; z < res; ++z) {
        for (int y = 0; y < res; ++y) {
            for (int x = 0; x < res; ++x) {
                // 1. 计算采样点
                Vec3 p = {
                    mesh.min_b.x + (x / (float)(res-1)) * size.x,
                    mesh.min_b.y + (y / (float)(res-1)) * size.y,
                    mesh.min_b.z + (z / (float)(res-1)) * size.z
                };

                // 2. 寻找最近三角形并计算绝对距离
                float min_d_sq = 1e10f;
                int closest_tri_idx = 0;
                for (size_t i = 0; i < mesh.indices.size(); i += 3) {
                    Vec3 dummy_n;
                    float d_sq = point_to_tri_dist_sq(p, mesh.vertices[mesh.indices[i]], mesh.vertices[mesh.indices[i+1]], mesh.vertices[mesh.indices[i+2]], dummy_n);
                    if (d_sq < min_d_sq) { min_d_sq = d_sq; closest_tri_idx = i; }
                }

                // 3. 核心：射线投票法判定符号 (inside/outside)
                int hit_count = 0;
                Vec3 ray_dir = {1.0f, 0.432f, 0.123f}; // 避开坐标轴的随机向量
                ray_dir = ray_dir * (1.0f / ray_dir.length());
                for (size_t i = 0; i < mesh.indices.size(); i += 3) {
                    if (ray_intersects_triangle(p, ray_dir, mesh.vertices[mesh.indices[i]], mesh.vertices[mesh.indices[i+1]], mesh.vertices[mesh.indices[i+2]])) {
                        hit_count++;
                    }
                }
                float dist = std::sqrt(min_d_sq);
                if (hit_count % 2 != 0) dist = -dist; // 奇数次穿越说明在内部

                // 4. 抓取最近顶点颜色 (棋盘格支持)
                uint32_t best_v = mesh.indices[closest_tri_idx];
                float best_vd = 1e10f;
                for(int k=0; k<3; k++) {
                    float d = (p - mesh.vertices[mesh.indices[closest_tri_idx+k]]).length_sq();
                    if(d < best_vd) { best_vd = d; best_v = mesh.indices[closest_tri_idx+k]; }
                }

                float ko;
                Color norm_c = normalize_to_255(mesh.colors[best_v], ko);
                voxels[x + y*res + z*res*res] = pack_voxel_v2(dist, pal.find_best_match(mesh.colors[best_v]), ko, world_size);
            }
        }
    }
    std::ofstream ofs(output_cem, std::ios::binary);
    ofs.write("CEM2", 4); ofs.write((char*)&res, 4);
    ofs.write((char*)&mesh.min_b, 12); ofs.write((char*)&mesh.max_b, 12);
    ofs.write((char*)voxels.data(), voxels.size() * 4);
}

// --- 升级版 Cook 函数 (64-bit) ---
void cook_cem_v3_64bit(const std::string& input_glb, const std::string& output_cem, GlobalPalette& pal) {
    MeshData mesh = load_glb(input_glb);
    int res = 64;
    std::vector<Voxel64> voxels(res * res * res);
    Vec3 size = mesh.max_b - mesh.min_b;
    float world_size = std::max({size.x, size.y, size.z});

    std::cout << "Cooking CEM3 (64-bit HD) with Normal Injection..." << std::endl;

    for (int z = 0; z < res; ++z) {
        for (int y = 0; y < res; ++y) {
            for (int x = 0; x < res; ++x) {
                Vec3 p = {
                    mesh.min_b.x + (x / (float)(res-1)) * size.x,
                    mesh.min_b.y + (y / (float)(res-1)) * size.y,
                    mesh.min_b.z + (z / (float)(res-1)) * size.z
                };

                // 1. 寻找最近三角形并计算重心
                float min_d_sq = 1e10f; int closest_tri = 0;
                float cu, cv, cw;
                for (size_t i = 0; i < mesh.indices.size(); i += 3) {
                    Vec3 dummy_n;
                    float d_sq = point_to_tri_dist_sq(p, mesh.vertices[mesh.indices[i]], mesh.vertices[mesh.indices[i+1]], mesh.vertices[mesh.indices[i+2]], dummy_n);
                    if (d_sq < min_d_sq) {
                        min_d_sq = d_sq; closest_tri = i;
                        get_closest_point_barycentric(p, mesh.vertices[mesh.indices[i]], mesh.vertices[mesh.indices[i+1]], mesh.vertices[mesh.indices[i+2]], cu, cv, cw);
                    }
                }

                // 2. 拿到平滑插值法线
                Vec3 n1 = mesh.normals[mesh.indices[closest_tri]];
                Vec3 n2 = mesh.normals[mesh.indices[closest_tri+1]];
                Vec3 n3 = mesh.normals[mesh.indices[closest_tri+2]];
                Vec3 smooth_n = (n1 * cu + n2 * cv + n3 * cw) * (1.0f / (cu + cv + cw + 1e-8f));
                smooth_n = smooth_n * (1.0f / smooth_n.length());

                // 3. 射线投票定正负 (保持 100% 准确)
                int hit_count = 0;
                Vec3 ray_dir = {1.0f, 0.432f, 0.123f};
                ray_dir = ray_dir * (1.0f / ray_dir.length());
                for (size_t i = 0; i < mesh.indices.size(); i += 3) {
                    if (ray_intersects_triangle(p, ray_dir, mesh.vertices[mesh.indices[i]], mesh.vertices[mesh.indices[i+1]], mesh.vertices[mesh.indices[i+2]])) hit_count++;
                }
                float dist = std::sqrt(min_d_sq);
                
                // Debug: 输出树干变绿位置的信息
                if (x == 32 && z == 32 && y == 10) {
                    printf("\n--- Voxel Brain Debug (%d,%d,%d) ---", x, y, z);
                    printf("Sample Point World Pos: (%f, %f, %f)\n", p.x, p.y, p.z);
                    
                    // 找到的最近三角形信息
                    Vec3 v1 = mesh.vertices[mesh.indices[closest_tri]];
                    Vec3 v2 = mesh.vertices[mesh.indices[closest_tri + 1]];
                    Vec3 v3 = mesh.vertices[mesh.indices[closest_tri + 2]];
                    
                    printf("Closest Triangle Vertices:\n");
                    printf("  V1: (%f, %f, %f)\n", v1.x, v1.y, v1.z);
                    printf("  V2: (%f, %f, %f)\n", v2.x, v2.y, v2.z);
                    printf("  V3: (%f, %f, %f)\n", v3.x, v3.y, v3.z);
                    
                    printf("Calculated Min Dist Sq: %f\n", min_d_sq);
                    
                    // 输出最近三角形所属的节点名称
                    int tri_idx = closest_tri / 3;
                    if (tri_idx >= 0 && tri_idx < (int)mesh.tri_sources.size()) {
                        printf("Closest Triangle belongs to Node: [%s]\n", mesh.tri_sources[tri_idx].node_name.c_str());
                    }
                }

                // --- 核心修正：针对细小几何体的保护 ---
                // 如果当前点离任何一个三角形的距离小于“半个体素”的对角线
                // 我们适当缩小 dist 的正值，让它在 3D 纹理中留下更明显的“痕迹”
                float voxel_world_size = world_size / 64.0f;
                if (hit_count % 2 != 0) {
                    dist = -dist;
                } else {
                    // 即使在外部，如果太近，也稍微压低距离值，防止插值时被抹除
                    if (dist < voxel_world_size * 0.5f) {
                        dist *= 0.8f;
                    }
                }

                // 4. 核心修复：插值 UV 并采样贴图
                Color raw_c = mesh.colors[mesh.indices[closest_tri]]; // 默认为第一个顶点的颜色
                if (mesh.texture_data && !mesh.uvs.empty()) {
                    // 1. 获取三个顶点的 UV
                    Vec2 uv1 = mesh.uvs[mesh.indices[closest_tri]];
                    Vec2 uv2 = mesh.uvs[mesh.indices[closest_tri + 1]];
                    Vec2 uv3 = mesh.uvs[mesh.indices[closest_tri + 2]];

                    // 2. 重心插值得到当前点的精确 UV
                    float u = uv1.u * cu + uv2.u * cv + uv3.u * cw;
                    float v = uv1.v * cu + uv2.v * cv + uv3.v * cw;

                    // 3. 映射到贴图像素坐标 (支持 Wrap 循环采样)
                    int tx = (int)(u * mesh.tex_w) % mesh.tex_w;
                    int ty = (int)(v * mesh.tex_h) % mesh.tex_h;
                    if (tx < 0) tx += mesh.tex_w;
                    if (ty < 0) ty += mesh.tex_h;

                    // 4. 从贴图内存读取
                    int pixel_idx = (ty * mesh.tex_w + tx) * 3;
                    raw_c.r = mesh.texture_data[pixel_idx];
                    raw_c.g = mesh.texture_data[pixel_idx + 1];
                    raw_c.b = mesh.texture_data[pixel_idx + 2];
                }

                // 5. 颜色与 Ko
                float ko;
                Color norm_c = normalize_to_255(raw_c, ko); // 获取原始 Ko

                // 6. 打包 64-bit
                voxels[x + y*res + z*res*res] = pack_voxel_64bit(dist, pal.find_best_match(raw_c), smooth_n, ko, world_size);
            }
        }
    }

    std::ofstream ofs(output_cem, std::ios::binary);
    ofs.write("CEM3", 4); // 魔数更新为 CEM3
    ofs.write((char*)&res, 4);
    ofs.write((char*)&mesh.min_b, 12); ofs.write((char*)&mesh.max_b, 12);
    ofs.write((char*)voxels.data(), voxels.size() * 8); // 写入 8 字节体素
    
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
            cook_cem_v3_64bit(input_glb, output_cem, pal);
        }
    }
    
    std::cout << "Cook-all completed!" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 4) return -1;
    std::string mode = argv[1];
    if (mode == "generate-palette") generate_global_palette(argv[2], argv[3]);
    else if (mode == "cook") { GlobalPalette pal; pal.load(argv[3]); cook_cem_v2(argv[2], argv[4], pal); }
    else if (mode == "cook-64bit") { GlobalPalette pal; pal.load(argv[3]); cook_cem_v3_64bit(argv[2], argv[4], pal); }
    else if (mode == "cook-all") {
        if (argc < 5) return -1;
        cook_all(argv[2], argv[3], argv[4]);
    }
    return 0;
}