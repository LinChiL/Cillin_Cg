import struct
import sys
import os
from collections import Counter

def decode_normal_10bit(packed):
    """还原 10-bit 八面体法线为 3D 向量"""
    ux = ((packed >> 5) & 0x1F) / 31.0
    uy = (packed & 0x1F) / 31.0
    x, y = ux * 2.0 - 1.0, uy * 2.0 - 1.0
    z = 1.0 - abs(x) - abs(y)
    
    if z < 0:
        x_old = x
        x = (1.0 - abs(y)) * (1.0 if x_old >= 0 else -1.0)
        y = (1.0 - abs(x_old)) * (1.0 if y >= 0 else -1.0)
    
    # 归一化
    mag = (x*x + y*y + z*z)**0.5
    return (x/mag, y/mag, z/mag) if mag > 0 else (0, 1, 0)

def inspect_cem3(cem_path, cpal_path=None):
    if not os.path.exists(cem_path):
        print(f"错误: 找不到文件 {cem_path}")
        return

    with open(cem_path, "rb") as f:
        # 1. 读取 32 字节头
        header = f.read(32)
        magic, res, min_x, min_y, min_z, max_x, max_y, max_z = struct.unpack("<4sIffffff", header)
        
        if magic != b"CEM3":
            print(f"错误: 该文件不是 CEM3 格式 (当前: {magic})")
            return

        print(f"--- CEM3 HD 资产报告: {os.path.basename(cem_path)} ---")
        print(f"版本: {magic.decode()}")
        print(f"分辨率: {res}x{res}x{res} (共 {res**3} 个体素)")
        print(f"AABB: ({min_x:.2f}, {min_y:.2f}, {min_z:.2f}) -> ({max_x:.2f}, {max_y:.2f}, {max_z:.2f})")

        # 2. 读取 Voxel 数据 (每个体素 8 字节: 2个 uint32)
        data = f.read()
        voxel_count = len(data) // 8
        raw_voxels = struct.unpack(f"<{voxel_count * 2}I", data)

        color_ids = []
        modifiers = []
        sdfs = []
        normals = []

        for i in range(0, len(raw_voxels), 2):
            r_chan = raw_voxels[i]     # [SDF 20][ColorID 12]
            g_chan = raw_voxels[i + 1] # [Normal 10][Ko 8][Reserved 14]

            # --- R通道解析 ---
            # ColorID: 低 12 位
            color_id = r_chan & 0xFFF
            color_ids.append(color_id)
            
            # SDF: 高 20 位 (处理补码)
            sdf_bits = (r_chan >> 12) & 0xFFFFF
            sdf_int = sdf_bits
            if sdf_bits >= 0x80000: # 2^19
                sdf_int -= 0x100000 # 2^20
            sdfs.append(sdf_int / 524287.0)

            # --- G通道解析 ---
            # Normal: 高 10 位 (32 - 10 = 22)
            norm_bits = (g_chan >> 22) & 0x3FF
            # Ko (Modifier): 位 14-21 (8位)
            mod_bits = (g_chan >> 14) & 0xFF
            
            modifiers.append(mod_bits / 255.0)
            # 只有在表面附近的体素才关心法线 (SDF绝对值小)
            if abs(sdfs[-1]) < 0.05:
                normals.append(decode_normal_10bit(norm_bits))

        # 3. 加载色板
        palette = []
        if cpal_path and os.path.exists(cpal_path):
            with open(cpal_path, "rb") as pf:
                pf.read(8)
                pal_raw = pf.read(1024 * 3)
                for j in range(0, len(pal_raw), 3):
                    palette.append(tuple(pal_raw[j:j+3]))

        # 4. 分析结果
        stats = Counter(color_ids)
        print("\n--- 12-bit 颜色分布 (Top 5) ---")
        for cid, count in stats.most_common(5):
            percent = (count / voxel_count) * 100
            color_str = f"RGB{palette[cid]}" if palette and cid < len(palette) else f"ID {cid}"
            print(f"[{color_str}]: 占比 {percent:5.1f}% ({count} 次)")

        print("\n--- 20-bit SDF 质量统计 ---")
        print(f"范围: [{min(sdfs):.4f} 到 {max(sdfs):.4f}]")
        inside_voxels = sum(1 for s in sdfs if s < 0)
        print(f"内部体素占比: {(inside_voxels/voxel_count)*100:.1f}%")

        print("\n--- 10-bit 法线 & Ko 统计 ---")
        avg_mod = sum(modifiers) / len(modifiers)
        print(f"平均亮度 (Ko): {avg_mod:.3f} (范围: {min(modifiers):.2f}-{max(modifiers):.2f})")
        
        if normals:
            # 采样一个典型的法线看看
            print(f"典型表面法线示例: {normals[len(normals)//2]}")

        # 5. 诊断建议
        print("\n--- 自动诊断建议 ---")
        # 如果颜色种类非常少，可能是 Cooker 没采到贴图
        if len(stats) < 10:
            print("警告: 检测到颜色种类过少，请确认 C++ Cooker 是否正确执行了重心坐标 UV 采样。")
        
        # 检查是否有非法 ID
        if any(cid >= 1024 for cid in stats.keys()):
            print("警告: 检测到超过 1024 的 Color ID，请检查全局色板分配逻辑。")

if __name__ == "__main__":
    default_cem_dir = "f:\\Cillin_CG\\Cillin_Cg\\Asset\\cemModel"
    default_cpal = "f:\\Cillin_CG\\Cillin_Cg\\Asset\\Global\\master.cpal"
    
    # 逻辑同你的旧脚本...
    if len(sys.argv) < 2:
        if os.path.exists(default_cem_dir):
            cem_files = [f for f in os.listdir(default_cem_dir) if f.endswith('.cem')]
            if cem_files:
                print(f"检测到 {len(cem_files)} 个 CEM3 文件:")
                for i, f in enumerate(cem_files): print(f"{i+1}. {f}")
                choice = int(input("\n序号: ")) - 1
                inspect_cem3(os.path.join(default_cem_dir, cem_files[choice]), default_cpal)
    else:
        inspect_cem3(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else default_cpal)