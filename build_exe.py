# -*- coding: utf-8 -*-
"""
智慧体育课堂监测系统 - PyInstaller 打包脚本

功能说明：
    将 client_app.py 打包成可分享的 Windows 可执行文件（exe）
    采用单文件模式 + 窗口模式，包含模型文件和隐式导入

使用方法：
    1. 确保已安装 PyInstaller：  pip install pyinstaller
    2. 确保已安装运行依赖：      pip install torch ultralytics mediapipe opencv-python PyQt5
    3. 运行本脚本：              python build_exe.py

作者：智慧体育课堂监测系统
"""

import os
import sys
import subprocess
import shutil

# ======================== 配置区域 ========================

# 应用名称（exe 文件名）
APP_NAME = "智慧体育课堂监测系统"

# 主程序入口脚本
ENTRY_SCRIPT = "client_app.py"

# 输出目录
DIST_DIR = "dist"

# 临时构建目录
BUILD_DIR = "build"

# 需要包含的数据文件列表（相对路径）
DATA_FILES = [
    "yolov8n-pose.pt",            # YOLOv8 姿态检测模型
    "blaze_pose_landmarker.task", # MediaPipe BlazePose 模型（可选）
]

# 需要隐式导入的模块（PyInstaller 无法自动检测到的依赖）
HIDDEN_IMPORTS = [
    "torch",
    "torchvision",
    "ultralytics",
    "mediapipe",
    "cv2",
    "PyQt5",
    "PyQt5.QtCore",
    "PyQt5.QtGui",
    "PyQt5.QtWidgets",
]


# ======================== 打包说明 ========================

def print_build_intro():
    """打印打包说明信息"""
    print("=" * 70)
    print("          智慧体育课堂监测系统 - PyInstaller 打包工具")
    print("=" * 70)
    print()
    print("【打包配置说明】")
    print(f"  - 应用名称    : {APP_NAME}")
    print(f"  - 主程序入口  : {ENTRY_SCRIPT}")
    print(f"  - 输出目录    : {DIST_DIR}/")
    print(f"  - 打包模式    : 单文件模式（--onefile）+ 窗口模式（--windowed）")
    print(f"  - 数据文件    : {', '.join(DATA_FILES)}")
    print(f"  - 隐式导入    : {', '.join(HIDDEN_IMPORTS)}")
    print()
    print("【环境要求】")
    print("  - Python 3.8 及以上版本")
    print("  - 已安装 PyInstaller：pip install pyinstaller")
    print("  - 已安装运行依赖库")
    print()
    print("【注意事项】")
    print("  - 首次打包可能需要较长时间（下载/分析依赖）")
    print("  - 打包过程中请勿关闭窗口")
    print("  - 生成的 exe 体积较大（包含模型和依赖库），属正常现象")
    print()
    print("=" * 70)
    print()
    print("开始打包...")
    print()


# ======================== 核心打包逻辑 ========================

def check_entry_script():
    """检查主程序入口脚本是否存在"""
    if not os.path.isfile(ENTRY_SCRIPT):
        print(f"[错误] 未找到主程序入口脚本：{ENTRY_SCRIPT}")
        print(f"       请确保 {ENTRY_SCRIPT} 与本脚本位于同一目录。")
        sys.exit(1)
    print(f"[检查] 主程序入口脚本 {ENTRY_SCRIPT} 存在。")


def collect_data_files():
    """
    收集实际存在的数据文件，构建 PyInstaller 的 --add-data 参数列表。

    返回值：
        list[str]: 形如 "src{sep}dest" 的参数字符串列表
    """
    data_args = []
    sep = ";" if sys.platform == "win32" else ":"

    for data_file in DATA_FILES:
        if os.path.isfile(data_file):
            # 文件存在，添加到打包数据中
            data_args.append(f"{data_file}{sep}.")
            print(f"[数据] 已包含数据文件：{data_file}")
        else:
            # 文件不存在（如 blaze_pose_landmarker.task 可选），跳过并提示
            print(f"[提示] 数据文件 {data_file} 不存在，已跳过（不影响打包）。")

    return data_args


def build_pyinstaller_command(data_args):
    """
    构建 PyInstaller 命令行参数列表。

    参数：
        data_args: 数据文件的 --add-data 参数列表

    返回值：
        list[str]: 完整的 PyInstaller 命令参数列表
    """
    cmd = [
        sys.executable,  # 使用当前 Python 解释器
        "-m",
        "PyInstaller",
        "--noconfirm",            # 覆盖已有输出，不询问确认
        "--clean",                # 清理 PyInstaller 缓存
        "--onefile",              # 单文件模式
        "--noconsole",            # 不显示控制台窗口
        "--windowed",             # 窗口模式（GUI 应用）
        "--name", APP_NAME,       # 输出应用名称
        "--distpath", DIST_DIR,   # 指定输出目录
        "--workpath", BUILD_DIR,  # 指定临时构建目录
        "--specpath", BUILD_DIR,  # 指定 spec 文件目录
    ]

    # 添加数据文件
    for data in data_args:
        cmd.extend(["--add-data", data])

    # 添加隐式导入
    for hidden in HIDDEN_IMPORTS:
        cmd.extend(["--hidden-import", hidden])

    # 主程序入口
    cmd.append(ENTRY_SCRIPT)

    return cmd


def run_build(cmd):
    """
    执行 PyInstaller 打包命令。

    参数：
        cmd: 完整的命令参数列表

    返回值：
        bool: 打包是否成功
    """
    print("[执行] PyInstaller 命令：")
    print("       " + " ".join(cmd))
    print()
    print("-" * 70)

    try:
        # 以子进程方式运行 PyInstaller
        result = subprocess.run(cmd, check=False)

        print("-" * 70)
        if result.returncode == 0:
            print("[成功] PyInstaller 打包过程执行完毕。")
            return True
        else:
            print(f"[失败] PyInstaller 执行出错，返回码：{result.returncode}")
            return False

    except FileNotFoundError:
        print("[错误] 未找到 PyInstaller，请先安装：pip install pyinstaller")
        return False
    except Exception as e:
        print(f"[错误] 打包过程中发生异常：{e}")
        return False


def verify_output():
    """
    验证打包输出文件是否存在。

    返回值：
        str or None: 输出 exe 的绝对路径；若不存在则返回 None
    """
    # Windows 下可执行文件扩展名为 .exe
    exe_name = f"{APP_NAME}.exe" if sys.platform == "win32" else APP_NAME
    exe_path = os.path.join(DIST_DIR, exe_name)

    if os.path.isfile(exe_path):
        return os.path.abspath(exe_path)
    return None


def print_success(output_path):
    """打印打包成功信息和输出路径"""
    print()
    print("=" * 70)
    print("                    打 包 成 功 ！")
    print("=" * 70)
    print()
    print(f"  应用名称    : {APP_NAME}")
    print(f"  输出文件    : {output_path}")
    print(f"  文件大小    : {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
    print()
    print("【分发说明】")
    print("  - 您可以直接将上述 exe 文件分享给其他 Windows 用户")
    print("  - 对方无需安装 Python 及任何依赖库即可运行")
    print("  - 首次启动可能稍慢（解压单文件所需），属正常现象")
    print()
    print("=" * 70)


# ======================== 主函数 ========================

def main():
    """主函数：协调打包流程"""
    # 1. 打印打包说明
    print_build_intro()

    # 2. 检查主程序入口
    check_entry_script()

    # 3. 收集数据文件
    data_args = collect_data_files()
    print()

    # 4. 构建 PyInstaller 命令
    cmd = build_pyinstaller_command(data_args)

    # 5. 执行打包
    success = run_build(cmd)

    # 6. 验证输出并提示
    if success:
        output_path = verify_output()
        if output_path:
            print_success(output_path)
        else:
            print()
            print("[警告] PyInstaller 执行完成，但未在输出目录中找到 exe 文件。")
            print(f"        请检查 {DIST_DIR}/ 目录，或查看上方的构建日志。")
            sys.exit(2)
    else:
        print()
        print("[失败] 打包未成功完成，请根据上方错误信息排查问题。")
        sys.exit(1)


if __name__ == "__main__":
    main()
