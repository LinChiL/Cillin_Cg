
# 在 python 目录下执行
#python inspect_cem.py <模型.cem> [色板.cpal]

# 例如：
#python inspect_cem.py ComeCube.cem master.cpal

import os

# 定义路径
project_dir = "f:\\Cillin_CG\\Cillin_Cg\\Code\\cillin_engine"
src_dir = os.path.join(project_dir, "src")
cpp_dir = os.path.join(project_dir, "cpp")
output_dir = "f:\\Cillin_CG\\Cillin_Cg\\Code\\cillin_engine\\python\\prompt"
output_file = os.path.join(output_dir, "prompt.txt")

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 获取所有rs文件，按路径排序（排除OLD文件夹）
files = []

# 处理Rust文件
for root, dirs, filenames in os.walk(src_dir):
    # 排除OLD文件夹
    if "OLD" in root:
        continue
    for file in filenames:
        if file.endswith((".rs", ".wgsl")):
            full_path = os.path.join(root, file)
            # 获取相对路径（相对于project_dir）
            rel_path = os.path.relpath(full_path, project_dir)
            files.append((rel_path, full_path))

# 处理特定的C++文件
if os.path.exists(cpp_dir):
    # 包含 cpp/cem 目录下的文件
    cpp_files = ["cem/cem_compiler_v4.cpp"]
    for file in cpp_files:
        full_path = os.path.join(cpp_dir, file)
        if os.path.exists(full_path):
            rel_path = os.path.relpath(full_path, project_dir)
            files.append((rel_path, full_path))
    
    # 包含 cpp/sdf 目录下的文件
    sdf_dir = os.path.join(cpp_dir, "sdf")
    if os.path.exists(sdf_dir):
        sdf_files = ["sdf_logic.cpp", "sdf_logic.hpp"]
        for file in sdf_files:
            full_path = os.path.join(sdf_dir, file)
            if os.path.exists(full_path):
                rel_path = os.path.relpath(full_path, project_dir)
                files.append((rel_path, full_path))

# 添加build.rs
build_rs_path = os.path.join(project_dir, "build.rs")
if os.path.exists(build_rs_path):
    files.append(("build.rs", build_rs_path))

# 按相对路径排序
files.sort(key=lambda x: x[0])

# 生成输出内容
output_lines = []
for rel_path, full_path in files:
    output_lines.append(rel_path)
    output_lines.append("")
    
    # 读取文件内容
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
            output_lines.append(content)
    except Exception as e:
        output_lines.append(f"// Error reading file: {e}")
    
    output_lines.append("")
    output_lines.append("")

# 写入文件
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("\n".join(output_lines))

print(f"Prompt file generated: {output_file}")
print(f"Total files processed: {len(files)}")
