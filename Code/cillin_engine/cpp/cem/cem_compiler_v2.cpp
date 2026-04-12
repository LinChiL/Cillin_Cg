#define CGLTF_IMPLEMENTATION
#include "cgltf.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <map>

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

struct MeshData {
    std::vector<Vec3> vertices;
    std::vector<Color> colors;
    std::vector<uint32_t> indices;
    Vec3 min_b, max_b;
};

// --- 色彩空间工具 ---
float color_dist(Color c1, Color c2) {
    return std::pow(c1.r - c2.r, 2) + std::pow(c1.g - c2.g, 2) + std::pow(c1.b - c2.b, 2);
}

class GlobalPalette {
public:
    std::vector<Color> colors;

    void load(const std::string& path) {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs.is_open()) {
            std::cout << "Warning: Palette not found, generating a default rainbow palette." << std::endl;
            generate_default();
            return;
        }
        colors.resize(1024);
        ifs.seekg(4); // 跳过魔数
        ifs.read((char*)colors.data(), 1024 * 3);
    }

    void generate_default() {
        colors.resize(1024);
        for(int i=0; i<1024; ++i) colors[i] = {(uint8_t)(i/4), (uint8_t)(i%255), (uint8_t)((i*7)%255)};
    }

    // 撞库逻辑：在 1024 种公用色里找最像的
    int find_best_match(Color target) {
        int best_id = 0;
        float min_d = 1e10f;
        for(int i=0; i<colors.size(); ++i) {
            float d = color_dist(target, colors[i]);
            if(d < min_d) { min_d = d; best_id = i; }
        }
        return best_id;
    }

    void save(const std::string& path) {
        std::ofstream ofs(path, std::ios::binary);
        ofs.write("CPAL", 4);
        uint32_t count = 1024;
        ofs.write((char*)&count, 4);
        ofs.write((char*)colors.data(), 1024 * 3);
        ofs.close();
        std::cout << "Saved palette to: " << path << std::endl;
    }
};

// --- 点到三角形距离函数 ---
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

// --- 核心：.cem v2 打包逻辑 ---
// [Dist: 12bit] [ColorID: 10bit] [Modifier: 8bit] [Flag: 2bit]
uint32_t pack_voxel_v2(float dist, int color_id, float modifier, float world_size) {
    // 1. 距离映射 (-2048 to 2047)
    float normalized_d = std::clamp(dist / world_size, -1.0f, 1.0f);
    int32_t d_int = (int32_t)(normalized_d * 2047.0f);
    uint32_t d_bits = (uint32_t)d_int & 0xFFF;

    // 2. 颜色 ID (0 to 1023)
    uint32_t id_bits = (uint32_t)color_id & 0x3FF;

    // 3. 亮度微调 (0 to 255)
    uint32_t mod_bits = (uint32_t)(std::clamp(modifier, 0.0f, 1.0f) * 255.0f);

    // 4. 组装
    return (d_bits << 20) | (id_bits << 10) | (mod_bits << 2) | 0u;
}

// --- 加载GLB模型 ---
MeshData load_glb(const std::string& path) {
    cgltf_options options = {};
    cgltf_data* data = nullptr;
    std::cout << "Loading GLB: " << path << std::endl;
    cgltf_result result = cgltf_parse_file(&options, path.c_str(), &data);
    if (result != cgltf_result_success) {
        std::cout << "Failed to parse GLB file. Error code: " << result << std::endl;
        exit(-1);
    }
    result = cgltf_load_buffers(&options, data, path.c_str());
    if (result != cgltf_result_success) {
        std::cout << "Failed to load buffers. Error code: " << result << std::endl;
        exit(-1);
    }

    MeshData mesh;
    mesh.min_b = {1e10, 1e10, 1e10}; mesh.max_b = {-1e10, -1e10, -1e10};

    for (size_t m_idx = 0; m_idx < data->meshes_count; ++m_idx) {
        cgltf_mesh* g_mesh = &data->meshes[m_idx];
        for (int i = 0; i < g_mesh->primitives_count; ++i) {
            cgltf_primitive* prim = &g_mesh->primitives[i];
            
            // --- 核心：处理贴图 ---
            uint8_t* img_data = nullptr;
            int img_w, img_h, img_comp;
            if (prim->material && prim->material->has_pbr_metallic_roughness && 
                prim->material->pbr_metallic_roughness.base_color_texture.texture) {
                
                auto* image = prim->material->pbr_metallic_roughness.base_color_texture.texture->image;
                auto* view = image->buffer_view;
                // 从内存中读取 GLB 内嵌的图片
                img_data = stbi_load_from_memory((uint8_t*)view->buffer->data + view->offset, view->size, &img_w, &img_h, &img_comp, 3);
            }

            // 获取当前 Primitive 的基础颜色（从材质球拿）
            Color mat_color = {255, 255, 255}; // 默认白
            if (prim->material && prim->material->has_pbr_metallic_roughness) {
                float* c = prim->material->pbr_metallic_roughness.base_color_factor;
                mat_color = { (uint8_t)(c[0] * 255), (uint8_t)(c[1] * 255), (uint8_t)(c[2] * 255) };
            }

            size_t vertex_offset = mesh.vertices.size();
            cgltf_accessor* pos_acc = nullptr;
            cgltf_accessor* uv_acc = nullptr;
            cgltf_accessor* col_acc = nullptr;

            for (int j = 0; j < prim->attributes_count; ++j) {
                if (prim->attributes[j].type == cgltf_attribute_type_position) pos_acc = prim->attributes[j].data;
                if (prim->attributes[j].type == cgltf_attribute_type_texcoord) uv_acc = prim->attributes[j].data;
                if (prim->attributes[j].type == cgltf_attribute_type_color) col_acc = prim->attributes[j].data;
            }

            if (pos_acc) {
                for (int k = 0; k < pos_acc->count; ++k) {
                    float v[3]; cgltf_accessor_read_float(pos_acc, k, v, 3);
                    Vec3 p = {v[0], v[1], v[2]};
                    mesh.vertices.push_back(p);
                    mesh.min_b.x = std::min(mesh.min_b.x, p.x); mesh.max_b.x = std::max(mesh.max_b.x, p.x);
                    mesh.min_b.y = std::min(mesh.min_b.y, p.y); mesh.max_b.y = std::max(mesh.max_b.y, p.y);
                    mesh.min_b.z = std::min(mesh.min_b.z, p.z); mesh.max_b.z = std::max(mesh.max_b.z, p.z);
                    
                    // 颜色逻辑：优先顶点色，其次贴图采样，最后材质颜色
                    Color final_c = mat_color;
                    if (col_acc) {
                        float c[4]; cgltf_accessor_read_float(col_acc, k, c, 4);
                        final_c = {(uint8_t)(c[0]*255), (uint8_t)(c[1]*255), (uint8_t)(c[2]*255)};
                    } else if (img_data && uv_acc) {
                        float uv[2]; cgltf_accessor_read_float(uv_acc, k, uv, 2);
                        int x = (int)(uv[0] * img_w) % img_w;
                        int y = (int)(uv[1] * img_h) % img_h;
                        if(x < 0) x += img_w;
                        if(y < 0) y += img_h;
                        int idx = (y * img_w + x) * 3;
                        final_c = {img_data[idx], img_data[idx+1], img_data[idx+2]};
                    }
                    mesh.colors.push_back(final_c);
                }
            }

            if (prim->indices) {
                for (int k = 0; k < prim->indices->count; ++k) {
                    mesh.indices.push_back(vertex_offset + cgltf_accessor_read_index(prim->indices, k));
                }
            }
            
            // 释放贴图数据
            if (img_data) {
                stbi_image_free(img_data);
            }
        }
    }
    cgltf_free(data);
    return mesh;
}

// --- 生成全局色板 ---
void generate_global_palette(const std::string& glb_dir, const std::string& output_cpal) {
    std::vector<std::string> glb_files = { "ComeCube.glb", "tree.glb" };
    std::map<uint32_t, double> color_weights; // 改用 double 存储权重

    // --- 策略 1：强行注入基础基因 (保底颜色) ---
    auto inject_color = [&](uint8_t r, uint8_t g, uint8_t b) {
        uint32_t key = (r << 16) | (g << 8) | b;
        color_weights[key] += 1000000.0; // 极高权重，确保入选
    };
    inject_color(255, 255, 255); // 纯白
    inject_color(0, 0, 0);       // 纯黑
    inject_color(128, 128, 128); // 中灰
    inject_color(255, 0, 255);   // 调试紫

    for (const auto& filename : glb_files) {
        std::string path = glb_dir + "\\" + filename;
        std::cout << "Processing: " << path << std::endl;
        MeshData mesh = load_glb(path);
        
        std::cout << "Mesh vertices: " << mesh.vertices.size() << ", colors: " << mesh.colors.size() << std::endl;
        
        // --- 策略 2：按比例加权 (防止高面数模型霸屏) ---
        // 每一个模型里出现的颜色，无论模型面数多高，都获得一个"固定选票"
        std::map<uint32_t, int> local_counts;
        for (const auto& c : mesh.colors) {
            uint32_t key = (c.r << 16) | (c.g << 8) | c.b;
            local_counts[key]++;
        }

        double model_weight_factor = 5000.0 / (double)mesh.colors.size();
        for (auto const& [key, count] : local_counts) {
            // 选票 = (该颜色在模型内的占比 * 固定权重) + 原始计数
            color_weights[key] += (count * model_weight_factor) + 1.0;
        }
    }

    std::vector<std::pair<uint32_t, double>> sorted_colors(color_weights.begin(), color_weights.end());
    std::sort(sorted_colors.begin(), sorted_colors.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    GlobalPalette pal;
    pal.colors.resize(1024, {0, 0, 0});
    for (int i = 0; i < std::min((size_t)1024, sorted_colors.size()); ++i) {
        uint32_t key = sorted_colors[i].first;
        pal.colors[i] = { (uint8_t)(key >> 16), (uint8_t)((key >> 8) & 0xFF), (uint8_t)(key & 0xFF) };
    }

    pal.save(output_cpal);
    std::cout << "Generated global palette with " << std::min((size_t)1024, sorted_colors.size()) << " colors" << std::endl;
}

// --- CEM 2.0 导出逻辑 ---
void cook_cem_v2(const std::string& input_glb, const std::string& output_cem, GlobalPalette& pal) {
    MeshData mesh = load_glb(input_glb);
    int res = 64;
    std::vector<uint32_t> voxels(res * res * res);
    Vec3 size = mesh.max_b - mesh.min_b;
    float world_size = std::max({size.x, size.y, size.z});

    for (int z = 0; z < res; ++z) {
        for (int y = 0; y < res; ++y) {
            for (int x = 0; x < res; ++x) {
                Vec3 p = {
                    mesh.min_b.x + (x / (float)(res-1)) * size.x,
                    mesh.min_b.y + (y / (float)(res-1)) * size.y,
                    mesh.min_b.z + (z / (float)(res-1)) * size.z
                };

                float min_d_sq = 1e10f;
                int closest_tri_idx = -1;
                Vec3 closest_n;

                for (size_t i = 0; i < mesh.indices.size(); i += 3) {
                    Vec3 a = mesh.vertices[mesh.indices[i]];
                    Vec3 b = mesh.vertices[mesh.indices[i+1]];
                    Vec3 c = mesh.vertices[mesh.indices[i+2]];
                    Vec3 tri_n;
                    float d_sq = point_to_tri_dist_sq(p, a, b, c, tri_n);
                    if (d_sq < min_d_sq) {
                        min_d_sq = d_sq;
                        closest_tri_idx = i;
                        closest_n = tri_n;
                    }
                }

                float dist = std::sqrt(min_d_sq);
                if ((p - mesh.vertices[mesh.indices[closest_tri_idx]]).dot(closest_n) < 0) dist = -dist;

                // --- 策略 3：精准颜色抓取 (不再只取第一个点) ---
                // 找到三角形里离采样点 p 最近的那个顶点
                uint32_t best_v_idx = mesh.indices[closest_tri_idx];
                float best_v_dist = 1e10f;
                for(int k=0; k<3; k++) {
                    float d = (p - mesh.vertices[mesh.indices[closest_tri_idx + k]]).length_sq();
                    if(d < best_v_dist) {
                        best_v_dist = d;
                        best_v_idx = mesh.indices[closest_tri_idx + k];
                    }
                }
                Color original_c = mesh.colors.empty() ? Color{255,255,255} : mesh.colors[best_v_idx];

                int color_id = pal.find_best_match(original_c);
                float mod = (original_c.r + original_c.g + original_c.b + 1.0f) /
                            (float)(pal.colors[color_id].r + pal.colors[color_id].g + pal.colors[color_id].b + 1.0f);

                voxels[x + y*res + z*res*res] = pack_voxel_v2(dist, color_id, mod, world_size);
            }
        }
    }
    // 写入文件逻辑保持不变 ...
    std::ofstream ofs(output_cem, std::ios::binary);
    ofs.write("CEM2", 4);
    ofs.write((char*)&res, 4);
    ofs.write((char*)&mesh.min_b, 12); ofs.write((char*)&mesh.max_b, 12);
    ofs.write((char*)voxels.data(), voxels.size() * 4);
    ofs.close();
    std::cout << "Cooked CEM v2: " << output_cem << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Usage: cem_cooker_v2 <mode> <input> <output>" << std::endl;
        std::cout << "Modes:" << std::endl;
        std::cout << "  generate-palette <glb_directory> <output.cpal>" << std::endl;
        std::cout << "  cook <input.glb> <palette.cpal> <output.cem>" << std::endl;
        return -1;
    }

    std::string mode = argv[1];

    if (mode == "generate-palette") {
        generate_global_palette(argv[2], argv[3]);
    } else if (mode == "cook") {
        if (argc < 5) {
            std::cout << "Usage: cem_cooker_v2 cook <input.glb> <palette.cpal> <output.cem>" << std::endl;
            return -1;
        }
        GlobalPalette pal;
        pal.load(argv[3]);
        cook_cem_v2(argv[2], argv[4], pal);
    } else {
        std::cout << "Unknown mode: " << mode << std::endl;
        return -1;
    }

    return 0;
}