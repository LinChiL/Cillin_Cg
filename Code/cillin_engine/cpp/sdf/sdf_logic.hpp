﻿#pragma once
#include "rust/cxx.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstdint>

struct Vec3 { float x, y, z; };

// 供 CPU 逻辑使用的采样逻辑（如碰撞检测）
float sample_local_sdf(Vec3 p_local, const uint32_t* data, int res);

// 全局 SDF 合并逻辑：将局部 SDF 数据合并到全局纹理
void merge_global_sdf(
    uint32_t* global_data,
    int global_res,
    const uint32_t* local_data,
    int local_res,
    float offset_x,
    float offset_y,
    float offset_z
);