﻿#include "sdf_logic.hpp"
#include "cillin_engine/src/lib.rs.h"

// 内部工具：仅解包距离，保持极简
inline float unpack_voxel_dist(uint32_t data) {
    uint32_t d_bits = (data >> 20) & 0xFFF;
    int32_t d_int = (int32_t)d_bits;
    if (d_bits >= 2048) d_int -= 4096;
    return (float)d_int / 2047.0f;
}

// 供 CPU 碰撞等逻辑使用的采样
float sample_local_sdf(Vec3 p_local, const uint32_t* data, int res) {
    // 映射到索引空间
    float x = (p_local.x * 0.5f + 0.5f) * (res - 1);
    float y = (p_local.y * 0.5f + 0.5f) * (res - 1);
    float z = (p_local.z * 0.5f + 0.5f) * (res - 1);

    if (x < 0 || x > res - 1 || y < 0 || y > res - 1 || z < 0 || z > res - 1) return 1.0f;

    int x0 = (int)x; int x1 = std::min(x0 + 1, res - 1);
    int y0 = (int)y; int y1 = std::min(y0 + 1, res - 1);
    int z0 = (int)z; int z1 = std::min(z0 + 1, res - 1);
    float fx = x - (float)x0; float fy = y - (float)y0; float fz = z - (float)z0;

    auto get_v = [&](int ix, int iy, int iz) {
        return unpack_voxel_dist(data[ix + iy * res + iz * res * res]);
    };

    // 这里 CPU 版使用 std::lerp 即可，不强制要求 Quintic
    return std::lerp(
        std::lerp(std::lerp(get_v(x0,y0,z0), get_v(x1,y0,z0), fx), std::lerp(get_v(x0,y1,z0), get_v(x1,y1,z0), fx), fy),
        std::lerp(std::lerp(get_v(x0,y0,z1), get_v(x1,y0,z1), fx), std::lerp(get_v(x0,y1,z1), get_v(x1,y1,z1), fx), fy),
        fz
    );
}

// merge_global_sdf 保持原有的逻辑即可
void merge_global_sdf(
    uint32_t* global_data,
    int global_res,
    const uint32_t* local_data,
    int local_res,
    float offset_x,
    float offset_y,
    float offset_z
) {
    // 计算局部 SDF 在全局纹理中的起始和结束坐标
    int start_x = static_cast<int>((offset_x + 0.5f) * (global_res - 1));
    int start_y = static_cast<int>((offset_y + 0.5f) * (global_res - 1));
    int start_z = static_cast<int>((offset_z + 0.5f) * (global_res - 1));
    
    int end_x = start_x + local_res;
    int end_y = start_y + local_res;
    int end_z = start_z + local_res;
    
    // 确保不越界
    start_x = std::max(0, start_x);
    start_y = std::max(0, start_y);
    start_z = std::max(0, start_z);
    
    end_x = std::min(global_res, end_x);
    end_y = std::min(global_res, end_y);
    end_z = std::min(global_res, end_z);
    
    // 合并数据
    for (int z = start_z; z < end_z; z++) {
        for (int y = start_y; y < end_y; y++) {
            for (int x = start_x; x < end_x; x++) {
                // 计算局部坐标
                int local_x = x - start_x;
                int local_y = y - start_y;
                int local_z = z - start_z;
                
                // 检查局部坐标是否在有效范围内
                if (local_x < 0 || local_x >= local_res ||
                    local_y < 0 || local_y >= local_res ||
                    local_z < 0 || local_z >= local_res) {
                    continue;
                }
                
                // 计算全局和局部索引
                int global_idx = x + y * global_res + z * global_res * global_res;
                int local_idx = local_x + local_y * local_res + local_z * local_res * local_res;
                
                // 直接复制数据
                global_data[global_idx] = local_data[local_idx];
            }
        }
    }
}