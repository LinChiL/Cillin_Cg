import os

def generate_directory_structure(root_dir, prefix=""):
    """生成目录结构字符串"""
    structure = []
    items = sorted(os.listdir(root_dir))
    
    for i, item in enumerate(items):
        path = os.path.join(root_dir, item)
        is_last = i == len(items) - 1
        
        if is_last:
            structure.append(f"{prefix}└── {item}")
            new_prefix = prefix + "    "
        else:
            structure.append(f"{prefix}├── {item}")
            new_prefix = prefix + "│   "
        
        if os.path.isdir(path):
            structure.extend(generate_directory_structure(path, new_prefix))
    
    return structure

def main():
    target_dir = r"f:\Cillin_CG\Cillin_Cg\Code\cillin_engine\src"
    output_file = r"f:\Cillin_CG\Cillin_Cg\Code\cillin_engine\python\directory_structure.txt"
    
    if not os.path.exists(target_dir):
        print(f"错误：目录 {target_dir} 不存在")
        return
    
    print(f"正在生成目录结构: {target_dir}")
    
    structure = generate_directory_structure(target_dir)
    structure_str = "\n".join(structure)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(structure_str)
    
    print(f"目录结构已保存到: {output_file}")
    print("\n目录结构预览:")
    print(structure_str)

if __name__ == "__main__":
    main()