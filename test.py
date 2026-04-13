
# 当前目录
current_dir = "/public/home/achwtr3mbb/wumr/UPTK-CUDA-test"

import os
import subprocess

# 支持的文件类型
EXTENSIONS = (".cpp", ".cu", ".h", ".hpp", ".c", ".cc")

def process_files(root_dir):
    for root, dirs, files in os.walk(root_dir):
        # 可选：跳过无关目录
        dirs[:] = [d for d in dirs if d not in (".git", "build", "out")]

        for file in files:
            if file.endswith(EXTENSIONS):
                input_abs = os.path.abspath(os.path.join(root, file))

                cmd = [
                    "python3",
                    "/public/home/achwtr3mbb/wumr/UPTK-CUDA-test/CUDA2UPTK.py",
                    "-i", input_abs,
                    "-o", input_abs   # ✅ 原地处理
                ]

                print("执行:", " ".join(cmd))

                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError:
                    print("❌ 失败:", input_abs)

# 当前目录
process_files(current_dir)
