#pragma once
#include "rust/cxx.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstdint> // 确保 uint32_t 可用

// 1. 全局命名空间的 Vec3 定义，供 cxx 使用
struct Vec3 {
    float x, y, z;
    Vec3 operator-(const Vec3& v) const { return {x - v.x, y - v.y, z - v.z}; }
    Vec3 operator+(const Vec3& v) const { return {x + v.x, y + v.y, z + v.z}; }
    float dot(const Vec3& v) const { return x * v.x + y * v.y + z * v.z; }
    Vec3 cross(const Vec3& v) const {
        return {y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x};
    }
    float length_sq() const { return this->dot(*this); }
};

// 2. 前置声明 Rust 中定义的结构体，这样编译器就知道它是一个结构体
struct EntitySdfInfo;

float point_to_triangle_dist_sq(Vec3 p, Vec3 a, Vec3 b, Vec3 c);
Vec3 transform_vec(Vec3 p, const float* matrix);
float unpack_voxel(uint32_t data);
float sample_local_sdf(Vec3 p_local, const uint32_t* data, int res);

// 2. 去掉 ffi:: 前缀
rust::Vec<float> generate_sdf_baked_aabb(
    rust::Slice<const float> vertices,
    rust::Slice<const uint32_t> indices,
    int res,
    float min_x, float min_y, float min_z,
    float max_x, float max_y, float max_z
);

rust::Vec<float> merge_global_sdf(
    rust::Slice<const EntitySdfInfo> entities, // 修改这里：去掉 ffi::
    int global_res,
    float world_box_size
);