﻿#define CGLTF_IMPLEMENTATION
#include "cgltf.h"
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

// --- 之前你提供的点到三角形距离函数 (保持并优化) ---
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

// --- 核心：.cem 打包逻辑 ---
// [Dist: 12bit] [ColorID: 10bit] [Modifier: 8bit] [Flag: 2bit]
uint32_t pack_voxel(float dist, int color_id, float modifier, float world_size) {
    // 距离映射：将相对于模型大小的距离映射到 12位 (-2048 to 2047)
    float normalized_d = std::clamp(dist / world_size, -1.0f, 1.0f);
    int32_t d_int = (int32_t)(normalized_d * 2047.0f);
    uint32_t d_bits = (uint32_t)d_int & 0xFFF;

    uint32_t id_bits = (uint32_t)color_id & 0x3FF;
    uint32_t mod_bits = (uint32_t)(std::clamp(modifier, 0.0f, 1.0f) * 255.0f);
    
    return (d_bits << 20) | (id_bits << 10) | (mod_bits << 2) | 0; // Flag 暂留 0
}

int main(int argc, char** argv) {
    if (argc < 3) { std::cout << "Usage: cem_cooker input.glb output.cem" << std::endl; return -1; }

    // 1. 加载 GLB
    cgltf_options options = {};
    cgltf_data* data = nullptr;
    if (cgltf_parse_file(&options, argv[1], &data) != cgltf_result_success) return -1;
    cgltf_load_buffers(&options, data, argv[1]);

    MeshData mesh;
    mesh.min_b = {1e10, 1e10, 1e10}; mesh.max_b = {-1e10, -1e10, -1e10};

    // 提取顶点、索引和颜色 (简化版，只读第一个 mesh)
    cgltf_mesh* g_mesh = &data->meshes[0];
    for (int i = 0; i < g_mesh->primitives_count; ++i) {
        cgltf_primitive* prim = &g_mesh->primitives[i];
        // 提取 Position
        for (int j = 0; j < prim->attributes_count; ++j) {
            if (prim->attributes[j].type == cgltf_attribute_type_position) {
                cgltf_accessor* acc = prim->attributes[j].data;
                for (int k = 0; k < acc->count; ++k) {
                    float v[3]; cgltf_accessor_read_float(acc, k, v, 3);
                    Vec3 p = {v[0], v[1], v[2]};
                    mesh.vertices.push_back(p);
                    mesh.min_b.x = std::min(mesh.min_b.x, p.x); mesh.max_b.x = std::max(mesh.max_b.x, p.x);
                    mesh.min_b.y = std::min(mesh.min_b.y, p.y); mesh.max_b.y = std::max(mesh.max_b.y, p.y);
                    mesh.min_b.z = std::min(mesh.min_b.z, p.z); mesh.max_b.z = std::max(mesh.max_b.z, p.z);
                }
            }
            // 提取 Color (如果有)
            if (prim->attributes[j].type == cgltf_attribute_type_color) {
                cgltf_accessor* acc = prim->attributes[j].data;
                for (int k = 0; k < acc->count; ++k) {
                    float c[4]; cgltf_accessor_read_float(acc, k, c, 4);
                    mesh.colors.push_back({(uint8_t)(c[0]*255), (uint8_t)(c[1]*255), (uint8_t)(c[2]*255)});
                }
            }
        }
        // 提取 Index
        for (int k = 0; k < prim->indices->count; ++k) {
            mesh.indices.push_back(cgltf_accessor_read_index(prim->indices, k));
        }
    }

    // 2. 自动生成颜色库 (Palette)
    std::vector<Color> palette;
    if (mesh.colors.empty()) palette.push_back({255, 255, 255}); // 默认白
    else {
        // 极简聚类：只存出现频率最高的前 1024 种颜色
        std::map<uint32_t, int> counts;
        for (auto c : mesh.colors) counts[(c.r << 16) | (c.g << 8) | c.b]++;
        for (auto const& [val, count] : counts) {
            palette.push_back({(uint8_t)(val >> 16), (uint8_t)(val >> 8), (uint8_t)(val & 0xFF)});
            if (palette.size() >= 1024) break;
        }
    }

    // 3. 开始烘焙全息网格
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
                Vec3 closest_n = {0,0,0};
                int closest_tri = 0;

                for (size_t i = 0; i < mesh.indices.size(); i += 3) {
                    Vec3 a = mesh.vertices[mesh.indices[i]];
                    Vec3 b = mesh.vertices[mesh.indices[i+1]];
                    Vec3 c = mesh.vertices[mesh.indices[i+2]];
                    Vec3 tri_n;
                    float d_sq = point_to_tri_dist_sq(p, a, b, c, tri_n);
                    if (d_sq < min_d_sq) { min_d_sq = d_sq; closest_n = tri_n; closest_tri = i; }
                }

                float dist = std::sqrt(min_d_sq);
                // Signed 判定
                if ((p - mesh.vertices[mesh.indices[closest_tri]]).dot(closest_n) < 0) dist = -dist;

                // 颜色采集与撞库
                Color original_c = mesh.colors.empty() ? Color{255,255,255} : mesh.colors[mesh.indices[closest_tri]];
                int best_id = 0; int min_diff = 1000;
                for (int i = 0; i < palette.size(); ++i) {
                    int diff = abs(palette[i].r - original_c.r) + abs(palette[i].g - original_c.g) + abs(palette[i].b - original_c.b);
                    if (diff < min_diff) { min_diff = diff; best_id = i; }
                }
                
                float mod = (original_c.r + original_c.g + original_c.b) / (float)(palette[best_id].r + palette[best_id].g + palette[best_id].b + 1);
                voxels[x + y*res + z*res*res] = pack_voxel(dist, best_id, mod, world_size);
            }
        }
    }

    // 4. 保存 .cem
    std::ofstream ofs(argv[2], std::ios::binary);
    ofs.write("CEM\1", 4);
    ofs.write((char*)&res, 4);
    ofs.write((char*)&mesh.min_b, 12); ofs.write((char*)&mesh.max_b, 12);
    // 写入 Palette (补齐 1024*3 字节)
    palette.resize(1024, {0,0,0});
    ofs.write((char*)palette.data(), 1024 * 3);
    ofs.write((char*)voxels.data(), voxels.size() * 4);
    ofs.close();

    std::cout << "Cooked success: " << argv[2] << " (Palette: " << palette.size() << " colors)" << std::endl;
    return 0;
}