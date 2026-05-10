import os
import sys
import numpy as np

# =========================
# 可调参数
# =========================

# 平移扰动幅度（单位通常为米）
TRANS_NOISE_STD = 0.01
TRANS_SMOOTH_WINDOW = 21

# 旋转扰动幅度（单位：度）
ROT_NOISE_STD_DEG = 0.9
ROT_SMOOTH_WINDOW = 21

# 输出序号起始值
START_INDEX = 11

# 四元数施加方式：
# "right": q_new = q_orig * dq
# "left" : q_new = dq * q_orig
ROTATION_MODE = "right"


# =========================
# 工具函数
# =========================
def moving_average(arr, window):
    if window <= 1:
        return arr.copy()

    if window % 2 == 0:
        window += 1

    pad = window // 2
    padded = np.pad(arr, ((pad, pad), (0, 0)), mode='edge')
    kernel = np.ones(window) / window

    out = np.zeros_like(arr)
    for d in range(arr.shape[1]):
        out[:, d] = np.convolve(padded[:, d], kernel, mode='valid')
    return out


def normalize_quaternion(q):
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return q / n


def quat_multiply(q1, q2):
    """
    四元数乘法，格式：[x, y, z, w]
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    return np.array([x, y, z, w], dtype=float)


def rotvec_to_quaternion(rv):
    """
    旋转向量 -> 四元数
    rv 长度为旋转角（弧度）
    输出格式：[x, y, z, w]
    """
    angle = np.linalg.norm(rv)
    if angle < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

    axis = rv / angle
    half = angle / 2.0
    s = np.sin(half)

    return normalize_quaternion(np.array([
        axis[0] * s,
        axis[1] * s,
        axis[2] * s,
        np.cos(half)
    ], dtype=float))


def read_trajectory_file(input_file):
    data = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            vals = [float(x) for x in line.split()]
            if len(vals) != 8:
                raise ValueError(
                    f"文件 {input_file} 第 {line_no} 行不是 8 列，而是 {len(vals)} 列：{line}"
                )
            data.append(vals)

    if len(data) == 0:
        raise ValueError(f"输入文件为空：{input_file}")

    return np.array(data, dtype=float)


def generate_perturbed_once(data, rng):
    """
    输入格式：
    timestamp tx ty tz qx qy qz qw
    """
    timestamps = data[:, 0]
    trans = data[:, 1:4]
    quats = data[:, 4:8]

    n = len(data)

    # 平移扰动
    raw_trans_noise = rng.randn(n, 3) * TRANS_NOISE_STD
    smooth_trans_noise = moving_average(raw_trans_noise, TRANS_SMOOTH_WINDOW)
    trans_new = trans + smooth_trans_noise

    # 旋转扰动
    rot_std_rad = np.deg2rad(ROT_NOISE_STD_DEG)
    raw_rotvec_noise = rng.randn(n, 3) * rot_std_rad
    smooth_rotvec_noise = moving_average(raw_rotvec_noise, ROT_SMOOTH_WINDOW)

    quats_new = np.zeros_like(quats)

    for i in range(n):
        q_orig = normalize_quaternion(quats[i])
        dq = rotvec_to_quaternion(smooth_rotvec_noise[i])

        if ROTATION_MODE.lower() == "left":
            q_new = quat_multiply(dq, q_orig)
        else:
            q_new = quat_multiply(q_orig, dq)

        quats_new[i] = normalize_quaternion(q_new)

    out = np.column_stack([timestamps, trans_new, quats_new])
    return out


def write_trajectory_file(output_file, data):
    with open(output_file, "w", encoding="utf-8") as f:
        for row in data:
            f.write(
                f"{row[0]:.6f} "
                f"{row[1]:.9f} {row[2]:.9f} {row[3]:.9f} "
                f"{row[4]:.9f} {row[5]:.9f} {row[6]:.9f} {row[7]:.9f}\n"
            )


def build_output_filename(input_file, index):
    folder = os.path.dirname(input_file)
    base = os.path.basename(input_file)
    stem, ext = os.path.splitext(base)

    if ext == "":
        ext = ".txt"

    output_name = f"{stem}_{index}{ext}"
    return os.path.join(folder, output_name)


def main():
    if len(sys.argv) != 3:
        print("用法：")
        print("  python3 perturb_trajectory_batch.py 输入文件名 生成次数")
        print("示例：")
        print("  python3 perturb_trajectory_batch.py CameraTrajectory.txt 10")
        sys.exit(1)

    input_file = sys.argv[1]

    try:
        num_copies = int(sys.argv[2])
    except ValueError:
        print("错误：生成次数必须是整数。")
        sys.exit(1)

    if num_copies <= 0:
        print("错误：生成次数必须大于 0。")
        sys.exit(1)

    if not os.path.isfile(input_file):
        print(f"错误：找不到输入文件：{input_file}")
        sys.exit(1)

    data = read_trajectory_file(input_file)
    total_lines = len(data)

    print(f"读取成功：{input_file}")
    print(f"总行数：{total_lines}")
    print(f"将生成 {num_copies} 个扰动文件，序号范围：{START_INDEX} 到 {START_INDEX + num_copies - 1}")

    for i in range(num_copies):
        output_index = START_INDEX + i

        # 每次用不同随机种子，保证每个输出文件都不同
        rng = np.random.RandomState(output_index)

        perturbed = generate_perturbed_once(data, rng)
        output_file = build_output_filename(input_file, output_index)
        write_trajectory_file(output_file, perturbed)

        print(f"已生成：{output_file}")

    print("全部完成。")


if __name__ == "__main__":
    main()
