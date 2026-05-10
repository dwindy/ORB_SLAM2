import os
import re
import sys
import subprocess

METRICS = ["max", "mean", "median", "min", "rmse", "sse", "std"]

def run_evo(gt_file, traj_file):
    cmd = ["evo_ape", "tum", gt_file, traj_file, "-a"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"命令执行失败: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    return result.stdout

def parse_metrics(output_text):
    stats = {}
    for line in output_text.splitlines():
        line = line.strip()
        m = re.match(r"^(max|mean|median|min|rmse|sse|std)\s+([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)$", line)
        if m:
            key = m.group(1)
            val = m.group(2)
            stats[key] = val
    return stats

def main():
    if len(sys.argv) != 5:
        print("用法：")
        print("  python3 batch_evo_ape.py groundtruth.txt 轨迹前缀 起始序号 次数")
        print("示例：")
        print("  python3 batch_evo_ape.py ~/Desktop/extract/Bonn/rgbd_bonn_balloon/groundtruth.txt CameraTrajectory 1 10")
        sys.exit(1)

    gt_file = os.path.expanduser(sys.argv[1])
    prefix = sys.argv[2]
    start_idx = int(sys.argv[3])
    count = int(sys.argv[4])

    if not os.path.isfile(gt_file):
        print(f"错误：groundtruth 文件不存在：{gt_file}")
        sys.exit(1)

    raw_output_file = "evo_ape_raw_output.txt"
    long_output_file = "evo_ape_stats_long.txt"
    matrix_output_file = "evo_ape_stats_matrix.txt"

    all_results = []

    for i in range(start_idx, start_idx + count):
        traj_file = f"{prefix}_{i:02d}.txt"

        if not os.path.isfile(traj_file):
            print(f"警告：文件不存在，跳过：{traj_file}")
            continue

        print(f"正在处理：{traj_file}")
        output = run_evo(gt_file, traj_file)
        stats = parse_metrics(output)

        all_results.append({
            "file": traj_file,
            "raw": output,
            "stats": stats
        })

    if len(all_results) == 0:
        print("没有成功处理任何轨迹文件。")
        sys.exit(1)

    with open(raw_output_file, "w", encoding="utf-8") as f:
        for item in all_results:
            f.write("=" * 80 + "\n")
            f.write(f"{item['file']}\n")
            f.write("=" * 80 + "\n")
            f.write(item["raw"])
            if not item["raw"].endswith("\n"):
                f.write("\n")
            f.write("\n")

    with open(long_output_file, "w", encoding="utf-8") as f:
        for item in all_results:
            f.write(f"{item['file']}\n")
            for metric in METRICS:
                value = item["stats"].get(metric, "")
                f.write(f"{metric}\t{value}\n")
            f.write("\n")

    with open(matrix_output_file, "w", encoding="utf-8") as f:
        header = ["metric"] + [item["file"] for item in all_results]
        f.write("\t".join(header) + "\n")
        for metric in METRICS:
            row = [metric]
            for item in all_results:
                row.append(item["stats"].get(metric, ""))
            f.write("\t".join(row) + "\n")

    print("\n完成。生成了 3 个文件：")
    print(f"1) {raw_output_file}")
    print(f"2) {long_output_file}")
    print(f"3) {matrix_output_file}")

if __name__ == "__main__":
    main()
