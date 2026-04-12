#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iostream>

struct Vec3 { float x, y, z; };
struct Color { uint8_t r, g, b; };

// 我们定义的 32 位全息数据包
// [Dist: 12bit] [ColorID: 10bit] [Modifier: 8bit] [Flag: 2bit]
struct VoxelPacked {
    uint32_t data;

    static uint32_t pack(float dist, int id, float mod, int flag) {
        // 1. 距离映射到 12位 (假设范围 -2.0 到 2.0)
        int16_t d = (int16_t)(std::clamp(dist / 2.0f, -1.0f, 1.0f) * 2047.0f);
        uint32_t d_bits = (uint32_t)d & 0xFFF; // 取低12位
        
        // 2. ID 占 10位 (0-1023)
        uint32_t id_bits = (uint32_t)id & 0x3FF;

        // 3. Modifier 占 8位 (0-255)
        uint32_t mod_bits = (uint32_t)(std::clamp(mod, 0.0f, 1.0f) * 255.0f);

        // 4. Flag 占 2位
        uint32_t flag_bits = (uint32_t)flag & 0x3;

        return (d_bits << 20) | (id_bits << 10) | (mod_bits << 2) | flag_bits;
    }
};

// 编译器核心类
class CemCompiler {
public:
    // 颜色库：自动将模型颜色归类
    std::vector<Color> palette;

    int find_or_add_palette(Color c) {
        // 简单的色彩空间距离匹配
        for(int i=0; i<palette.size(); ++i) {
            auto p = palette[i];
            int diff = abs(p.r - c.r) + abs(p.g - c.g) + abs(p.b - c.b);
            if(diff < 10) return i; // 足够接近，撞库成功
        }
        if(palette.size() < 1024) {
            palette.push_back(c);
            return palette.size() - 1;
        }
        return 0; // 库满，强行归类到第一个
    }

    void compile(const std::string& glb_data, const std::string& output_path) {
        int res = 64;
        std::vector<uint32_t> voxel_grid(res * res * res);

        // --- 核心计算循环 ---
        for (int z = 0; z < res; ++z) {
            for (int y = 0; y < res; ++y) {
                for (int x = 0; x < res; ++x) {
                    Vec3 p = calculate_grid_pos(x, y, z);
                    
                    // 1. 计算 Signed Distance
                    float dist = calculate_signed_distance(p);
                    
                    // 2. 抓取该点在 Mesh 上的原始颜色
                    Color mesh_color = sample_mesh_color(p);
                    
                    // 3. 颜色降维：寻找最匹配的库 ID 和 亮度微调值
                    int id = find_or_add_palette(mesh_color);
                    float modifier = calculate_brightness_mod(mesh_color, palette[id]);
                    
                    // 4. 打包进入 32bit 抽屉
                    voxel_grid[x + y*res + z*res*res] = VoxelPacked::pack(dist, id, modifier, 0);
                }
            }
        }

        // --- 写入 .cem 文件 ---
        std::ofstream ofs(output_path, std::ios::binary);
        ofs.write("CEM\0", 4);
        // 写入 AABB, Palette, 然后是 Voxel Data
        ofs.write((char*)voxel_grid.data(), voxel_grid.size() * 4);
        ofs.close();
        std::cout << "Successfully cooked: " << output_path << std::endl;
    }

private:
    // 这里的逻辑就是我们之前讨论的重心坐标插值和 AABB 映射
    float calculate_brightness_mod(Color original, Color palette_base) {
        float o_lum = (original.r + original.g + original.b) / 3.0f;
        float p_lum = (palette_base.r + palette_base.g + palette_base.b) / 3.0f;
        if (p_lum < 1.0f) return 1.0f;
        return std::clamp(o_lum / p_lum, 0.0f, 1.0f);
    }
};