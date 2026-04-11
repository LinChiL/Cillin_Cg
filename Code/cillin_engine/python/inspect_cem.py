import struct
import sys
import os
from collections import Counter

def inspect_cem(cem_path, cpal_path=None):
    if not os.path.exists(cem_path):
        print(f"错误: 找不到文件 {cem_path}")
        return

    with open(cem_path, "rb") as f:
        # 1. 读取 32 字节头
        header = f.read(32)
        magic, res, min_x, min_y, min_z, max_x, max_y, max_z = struct.unpack("<4sIffffff", header)
        
        if magic != b"CEM2":
            print(f"错误: 不支持的格式 {magic}")
            return

        print(f"--- CEM 资产报告: {os.path.basename(cem_path)} ---")
        print(f"版本: {magic.decode()}")
        print(f"分辨率: {res}x{res}x{res} (共 {res**3} 个格点)")
        print(f"AABB Min: ({min_x:.2f}, {min_y:.2f}, {min_z:.2f})")
        print(f"AABB Max: ({max_x:.2f}, {max_y:.2f}, {max_z:.2f})")

        # 2. 读取 Voxel 数据
        data = f.read()
        voxels = struct.unpack(f"<{len(data)//4}I", data)

        # 3. 统计颜色 ID
        color_ids = []
        for v in voxels:
            # 根据你的 pack_voxel_v2 逻辑：[Dist:12][ColorID:10][Mod:8][Flag:2]
            # ID 在位 10-19
            color_id = (v >> 10) & 0x3FF
            color_ids.append(color_id)

        # 4. 如果提供了色板，加载颜色
        palette = []
        if cpal_path and os.path.exists(cpal_path):
            with open(cpal_path, "rb") as pf:
                pf.read(8) # 跳过头
                pal_raw = pf.read(1024 * 3)
                for i in range(0, len(pal_raw), 3):
                    palette.append(tuple(pal_raw[i:i+3]))

        # 5. 分析结果
        stats = Counter(color_ids)
        print("\n--- 颜色分布分析 (Top 10) ---")
        total = len(color_ids)
        for cid, count in stats.most_common(10):
            percent = (count / total) * 100
            color_str = f"RGB{palette[cid]}" if palette and cid < len(palette) else "未知"
            print(f"ID {cid:4}: 出现 {count:8} 次 ({percent:5.1f}%) | 对应颜色: {color_str}")

        # 6. 验证棋盘格特征
        if len(stats) >= 2:
            print("\n结论: 文件包含多种颜色 ID，数据【正常】。如果引擎里是纯色，则是渲染逻辑或索引问题。")
        else:
            print("\n结论: 文件只包含一种颜色 ID，数据【异常】。请检查 C++ Compiler 导出逻辑。")

if __name__ == "__main__":
    # 默认路径
    default_cem_dir = "f:\\Cillin_CG\\Cillin_Cg\\Asset\\cemModel"
    default_cpal = "f:\\Cillin_CG\\Cillin_Cg\\Asset\\Global\\master.cpal"
    
    if len(sys.argv) < 2:
        # 自动检测目录中的 .cem 文件
        if os.path.exists(default_cem_dir):
            cem_files = [f for f in os.listdir(default_cem_dir) if f.endswith('.cem')]
            if cem_files:
                print(f"自动检测到 {len(cem_files)} 个 CEM 文件:")
                for i, f in enumerate(cem_files):
                    print(f"{i+1}. {f}")
                
                # 等待用户输入
                try:
                    choice = int(input("\n请输入要检查的文件序号: ")) - 1
                    if 0 <= choice < len(cem_files):
                        selected_file = cem_files[choice]
                        m_path = os.path.join(default_cem_dir, selected_file)
                        p_path = default_cpal
                        print(f"\n正在检查: {selected_file}")
                        inspect_cem(m_path, p_path)
                    else:
                        print("错误: 输入的序号无效")
                except ValueError:
                    print("错误: 请输入有效的数字")
            else:
                print(f"错误: 在 {default_cem_dir} 中未找到 .cem 文件")
        else:
            print(f"错误: 目录 {default_cem_dir} 不存在")
    else:
        m_path = sys.argv[1]
        p_path = sys.argv[2] if len(sys.argv) > 2 else default_cpal
        inspect_cem(m_path, p_path)
