#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

# ============================================================================
# 配置区域：选择要处理的序列（只处理下面列表中的名称，可注释）
# 若要处理全部，设置 PROCESS_ALL = True
# ============================================================================
PROCESS_ALL = False   # 若为 True，则处理所有 rgbd_dataset_* 子目录（忽略 SEQUENCES 列表）

# 手动指定要处理的序列（子目录名称，包含 rgbd_dataset_ 前缀）
# 不需要的可以注释掉或删除
SEQUENCES = [
    "rgbd_dataset_freiburg2_desk_with_person",
    "rgbd_dataset_freiburg3_sitting_halfsphere",
    "rgbd_dataset_freiburg3_sitting_rpy",
    "rgbd_dataset_freiburg3_sitting_static",
    "rgbd_dataset_freiburg3_sitting_xyz",
    "rgbd_dataset_freiburg3_walking_halfsphere",
    "rgbd_dataset_freiburg3_walking_rpy",
    "rgbd_dataset_freiburg3_walking_static",
    "rgbd_dataset_freiburg3_walking_xyz",
]
# ============================================================================

def main():
    script_dir = Path(__file__).parent.resolve()   # TUM 目录
    os.chdir(script_dir)

    # 获取所有以 rgbd_dataset_ 开头的子目录
    all_seq_dirs = [d for d in script_dir.iterdir() if d.is_dir() and d.name.startswith("rgbd_dataset_")]
    if not all_seq_dirs:
        print("未找到任何 rgbd_dataset_* 子目录")
        return

    # 确定要处理的目录列表
    if PROCESS_ALL:
        target_dirs = all_seq_dirs
        print(f"处理所有 {len(target_dirs)} 个序列目录")
    else:
        # 仅处理 SEQUENCES 中列出的且实际存在的目录
        target_dirs = [script_dir / name for name in SEQUENCES if (script_dir / name).is_dir()]
        if not target_dirs:
            print("未找到任何匹配的序列（请检查 SEQUENCES 列表中的名称是否正确，或设置 PROCESS_ALL = True）")
            return
        print(f"将处理以下 {len(target_dirs)} 个序列:")
        for d in target_dirs:
            print(f"  - {d.name}")

    # 对每个选中的序列执行处理
    for seq_dir in target_dirs:
        print(f"\n--- 进入目录: {seq_dir.name} ---")
        os.chdir(seq_dir)

        # 构建 groundtruth 路径
        gt_file = Path.home() / "Downloads" / "dataset" / "TUM" / seq_dir.name / "groundtruth.txt"

        # 调用上一级目录（TUM）中的 batch_evo_ape.py
        script_path = "../batch_evo_ape.py"
        cmd = ["python3", str(script_path), str(gt_file), "CameraTrajectory", "1", "10"]

        print(f"执行命令: {' '.join(map(str, cmd))}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"错误: 序列 {seq_dir.name} 处理失败，退出码 {e.returncode}")
            # 继续处理下一个序列
        finally:
            os.chdir(script_dir)   # 返回 TUM 目录

    print("\n所有选中的序列处理完成。")

if __name__ == "__main__":
    main()
