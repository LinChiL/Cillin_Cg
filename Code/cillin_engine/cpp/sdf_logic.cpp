#include "sdf_logic.hpp"
#include "cillin_engine/src/lib.rs.h" // 必须包含这个！cxx 会自动生成它

float point_to_triangle_dist_sq(Vec3 p, Vec3 a, Vec3 b, Vec3 c) {
    Vec3 ab = b - a;
    Vec3 bc = c - b;
    Vec3 ca = a - c;
    Vec3 ap = p - a;
    Vec3 bp = p - b;
    Vec3 cp = p - c;

    Vec3 n = ab.cross(ca);

    auto project_on_segment = [](Vec3 p, Vec3 a, Vec3 b) {
        Vec3 ab = b - a;
        float t = (p - a).dot(ab) / ab.length_sq();
        t = std::max(0.0f, std::min(1.0f, t));
        Vec3 proj = {a.x + t * ab.x, a.y + t * ab.y, a.z + t * ab.z};
        Vec3 diff = p - proj;
        return diff.length_sq();
    };

    bool inside = 
        ab.cross(n).dot(ap) >= 0.0f && 
        bc.cross(n).dot(bp) >= 0.0f && 
        ca.cross(n).dot(cp) >= 0.0f;

    if (inside) {
        float dist_n = n.dot(ap);
        return (dist_n * dist_n) / n.length_sq();
    } else {
        return std::min({
            project_on_segment(p, a, b),
            project_on_segment(p, b, c),
            project_on_segment(p, c, a)
        });
    }
}

rust::Vec<float> generate_sdf_baked_aabb(
    rust::Slice<const float> vertices,
    rust::Slice<const uint32_t> indices,
    int res,
    float min_x, float min_y, float min_z,
    float max_x, float max_y, float max_z
) {
    rust::Vec<float> output_grid;
    output_grid.reserve(res * res * res);

    Vec3 min_b = { min_x, min_y, min_z };
    Vec3 max_b = { max_x, max_y, max_z };
    Vec3 size = { max_x - min_x, max_y - min_y, max_z - min_z };

    for (int z = 0; z < res; ++z) {
        for (int y = 0; y < res; ++y) {
            for (int x = 0; x < res; ++x) {
                // 精准映射：根据格子索引算出世界坐标
                Vec3 p = {
                    min_b.x + (x / (float)(res - 1)) * size.x,
                    min_b.y + (y / (float)(res - 1)) * size.y,
                    min_b.z + (z / (float)(res - 1)) * size.z
                };

                float min_d_sq = 1e10f;
                Vec3 closest_normal = {0, 0, 0};
                Vec3 closest_p_to_v = {0, 0, 0};

                for (size_t i = 0; i < indices.size(); i += 3) {
                    Vec3 a = { vertices[indices[i]*8], vertices[indices[i]*8+1], vertices[indices[i]*8+2] };
                    Vec3 b = { vertices[indices[i+1]*8], vertices[indices[i+1]*8+1], vertices[indices[i+1]*8+2] };
                    Vec3 c = { vertices[indices[i+2]*8], vertices[indices[i+2]*8+1], vertices[indices[i+2]*8+2] };
                    
                    // 计算该三角形的法线
                    Vec3 edge1 = b - a;
                    Vec3 edge2 = c - a;
                    Vec3 tri_normal = edge1.cross(edge2); // 未归一化的法线

                    float d_sq = point_to_triangle_dist_sq(p, a, b, c);
                    if (d_sq < min_d_sq) {
                        min_d_sq = d_sq;
                        closest_normal = tri_normal;
                        closest_p_to_v = p - a; // 从面上一点指向采样点 p 的向量
                    }
                }

                float dist = std::sqrt(min_d_sq);
                
                // --- 核心：判定符号 ---
                // 如果点 p 到面的向量与法线方向相反（点积为负），说明在模型内部
                if (closest_p_to_v.dot(closest_normal) < 0) {
                    dist = -dist; // 标记为内部！
                }
                
                output_grid.push_back(dist);
            }
        }
    }
    return output_grid;
}

// 辅助函数：矩阵变换点
Vec3 transform_vec(Vec3 p, const float* matrix) {
    float x = p.x * matrix[0] + p.y * matrix[4] + p.z * matrix[8] + matrix[12];
    float y = p.x * matrix[1] + p.y * matrix[5] + p.z * matrix[9] + matrix[13];
    float z = p.x * matrix[2] + p.y * matrix[6] + p.z * matrix[10] + matrix[14];
    float w = p.x * matrix[3] + p.y * matrix[7] + p.z * matrix[11] + matrix[15];
    
    if (w != 0.0f) {
        return {x / w, y / w, z / w};
    }
    return {x, y, z};
}

// 解包体素数据
float unpack_voxel(uint32_t data) {
    // 手动处理 12 位补码，确保负数正确还原
    uint32_t d_bits = data >> 20;
    int32_t d_int = (int32_t)d_bits;
    if (d_bits >= 2048) {
        d_int -= 4096;
    }
    return (float)d_int / 2047.0f;
}

// 三线性插值采样（支持 Uint32 全息数据）
float sample_local_sdf(Vec3 p_local, const uint32_t* data, int res) {
    // 将 -1~1 的局部坐标转为 0~res-1 的索引坐标
    float x = (p_local.x * 0.5f + 0.5f) * (res - 1);
    float y = (p_local.y * 0.5f + 0.5f) * (res - 1);
    float z = (p_local.z * 0.5f + 0.5f) * (res - 1);

    // 如果坐标在模型包围盒外，返回一个很大的距离
    if (x < 0 || x > res-1 || y < 0 || y > res-1 || z < 0 || z > res-1) return 10.0f;

    int x0 = (int)x; int x1 = std::min(x0 + 1, res - 1);
    int y0 = (int)y; int y1 = std::min(y0 + 1, res - 1);
    int z0 = (int)z; int z1 = std::min(z0 + 1, res - 1);

    float fx = x - x0; float fy = y - y0; float fz = z - z0;

    // 8 点采样插值 (标准的 GPU 纹理采样算法的 CPU 实现)
    auto get_v = [&](int ix, int iy, int iz) { 
        return unpack_voxel(data[ix + iy * res + iz * res * res]); 
    };

    float v000 = get_v(x0, y0, z0); float v100 = get_v(x1, y0, z0);
    float v010 = get_v(x0, y1, z0); float v110 = get_v(x1, y1, z0);
    float v001 = get_v(x0, y0, z1); float v101 = get_v(x1, y0, z1);
    float v011 = get_v(x0, y1, z1); float v111 = get_v(x1, y1, z1);

    float i1 = v000 * (1 - fx) + v100 * fx;
    float i2 = v010 * (1 - fx) + v110 * fx;
    float j1 = v001 * (1 - fx) + v101 * fx;
    float j2 = v011 * (1 - fx) + v111 * fx;

    float w1 = i1 * (1 - fy) + i2 * fy;
    float w2 = j1 * (1 - fy) + j2 * fy;

    return w1 * (1 - fz) + w2 * fz;
}

// 核心：合并所有实体到全局 SDF
rust::Vec<float> merge_global_sdf(
    rust::Slice<const EntitySdfInfo> entities, // 修改这里：去掉 ffi::
    int global_res,
    float world_box_size
) {
    rust::Vec<float> global_grid;
    global_grid.reserve(global_res * global_res * global_res);

    float step = world_box_size / (float)global_res;
    float offset = world_box_size * 0.5f;
    float half_step = step * 0.5f;

    for (int z = 0; z < global_res; ++z) {
        for (int y = 0; y < global_res; ++y) {
            for (int x = 0; x < global_res; ++x) {
                Vec3 p_world = { 
                    x * step - offset + half_step, 
                    y * step - offset + half_step, 
                    z * step - offset + half_step 
                };
                float min_dist = 10.0f;

                for (const auto& entity : entities) {
                    // 使用 entity.inv_matrix.data() 访问数组
                    Vec3 p_local = transform_vec(p_world, entity.inv_matrix.data());
                    float d = sample_local_sdf(p_local, entity.sdf_ptr, entity.res);
                    if (d < min_dist) min_dist = d;
                }
                global_grid.push_back(min_dist);
            }
        }
    }
    return global_grid;
}