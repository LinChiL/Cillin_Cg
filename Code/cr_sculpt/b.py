import os

src_dir = os.path.join(os.path.dirname(__file__), "src")
output_file = os.path.join(os.path.dirname(__file__), "packed_code.txt")

extensions = {".rs", ".wgsl", ".toml", ".json"}

with open(output_file, "w", encoding="utf-8") as out:
    for root, _, files in os.walk(src_dir):
        for file in sorted(files):
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, src_dir)
                out.write(f"===== {rel_path} =====\n")
                with open(file_path, "r", encoding="utf-8") as f:
                    out.write(f.read())
                out.write("\n\n")