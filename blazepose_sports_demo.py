# -*- coding: utf-8 -*-
"""
校园智慧体育课堂 - BlazePose 危险动作监测 Demo（MediaPipe Tasks API 版）
========================================================================
功能：
  1. 实时人体 33 关键点姿态估计（BlazePose Lite）
  2. 危险动作规则库：弯腰过猛 / 膝盖过脚尖 / 屈膝过度 / 膝盖内扣
  3. 课堂状态识别：站立 / 运动 / 静止
  4. 实时告警与可视化
  5. 支持摄像头或本地视频文件输入，可选保存结果视频

用法：
  摄像头实时监测：  python blazepose_sports_demo.py
  指定视频文件：    python blazepose_sports_demo.py --input test.mp4
  保存结果视频：    python blazepose_sports_demo.py --input test.mp4 --output out.mp4
  指定模型路径：    python blazepose_sports_demo.py --model models/pose_landmarker_lite.task

依赖：mediapipe>=0.10.30 opencv-python numpy
模型：pose_landmarker_lite.task（首次运行自动从官方下载，或用 --model 指定）
"""

import argparse
import math
import os
import time
import urllib.request
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ============== BlazePose 33 关键点索引 ==============
NOSE = 0
L_SHO, R_SHO = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_FOOT, R_FOOT = 31, 32

# 骨架连接关系（BlazePose 33 点）
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"


def ensure_model(model_path):
    """确保模型文件存在，不存在则下载"""
    if model_path and os.path.exists(model_path):
        return model_path
    default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "models", "pose_landmarker_lite.task")
    if os.path.exists(default_path):
        return default_path
    os.makedirs(os.path.dirname(default_path), exist_ok=True)
    print(f"[下载] 正在下载 BlazePose 模型到 {default_path} ...")
    urllib.request.urlretrieve(MODEL_URL, default_path)
    print("[下载] 完成")
    return default_path


# ============== 几何工具函数 ==============
def to_pixel(lm, w, h):
    """归一化坐标 -> 像素坐标"""
    return int(lm.x * w), int(lm.y * h)


def angle_3pts(a, b, c):
    """三点夹角（度），b 为顶点"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_v = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return math.degrees(math.acos(np.clip(cos_v, -1.0, 1.0)))


def midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def trunk_forward_angle(shoulder_mid, hip_mid):
    """躯干前倾角：0=直立，90=水平前倾"""
    dx = shoulder_mid[0] - hip_mid[0]
    dy = shoulder_mid[1] - hip_mid[1]
    return math.degrees(math.atan2(abs(dx), abs(dy) + 1e-8))


# ============== 危险动作规则库 ==============
THRESH = {
    "trunk_forward_warn": 55,
    "trunk_forward_danger": 75,
    "knee_over_toe_margin": 0.02,
    "squat_knee_open_min": 150,
    "valgus_ratio": 0.04,
}


def detect_risks(lm, w, h):
    """输入单帧关键点列表，返回 (风险列表, 指标字典)"""
    risks = []
    if lm is None:
        return risks, {"trunk_ang": 0, "knee_ang_min": 0, "l_knee_ang": 0, "r_knee_ang": 0}

    l_sho = to_pixel(lm[L_SHO], w, h)
    r_sho = to_pixel(lm[R_SHO], w, h)
    l_hip = to_pixel(lm[L_HIP], w, h)
    r_hip = to_pixel(lm[R_HIP], w, h)
    l_knee = to_pixel(lm[L_KNEE], w, h)
    r_knee = to_pixel(lm[R_KNEE], w, h)
    l_ankle = to_pixel(lm[L_ANKLE], w, h)
    r_ankle = to_pixel(lm[R_ANKLE], w, h)

    sho_mid = midpoint(l_sho, r_sho)
    hip_mid = midpoint(l_hip, r_hip)

    # 1) 弯腰过猛
    trunk_ang = trunk_forward_angle(sho_mid, hip_mid)
    if trunk_ang >= THRESH["trunk_forward_danger"]:
        risks.append(("danger", f"弯腰过猛({trunk_ang:.0f}°)"))
    elif trunk_ang >= THRESH["trunk_forward_warn"]:
        risks.append(("warn", f"躯干前倾({trunk_ang:.0f}°)"))

    # 2) 膝盖过脚尖（仅在膝关节实际弯曲时检查，避免站立误报）
    l_knee_ang_pre = angle_3pts(l_hip, l_knee, l_ankle)
    r_knee_ang_pre = angle_3pts(r_hip, r_knee, r_ankle)
    for side, knee_i, foot_i, kang in [("左", L_KNEE, L_FOOT, l_knee_ang_pre),
                                        ("右", R_KNEE, R_FOOT, r_knee_ang_pre)]:
        if kang < 160 and lm[knee_i].x - lm[foot_i].x > THRESH["knee_over_toe_margin"]:
            risks.append(("warn", f"{side}膝过脚尖"))

    # 3) 膝关节角（复用第2步的计算）
    l_knee_ang, r_knee_ang = l_knee_ang_pre, r_knee_ang_pre
    knee_ang_min = min(l_knee_ang, r_knee_ang)
    if knee_ang_min < THRESH["squat_knee_open_min"]:
        risks.append(("warn", f"屈膝过度({knee_ang_min:.0f}°)"))

    # 4) 膝盖内扣
    def valgus_offset(hip, knee, ankle):
        hip, knee, ankle = np.array(hip), np.array(knee), np.array(ankle)
        line = ankle - hip
        norm = np.linalg.norm(line) + 1e-8
        normal = np.array([-line[1], line[0]]) / norm
        return np.dot(knee - hip, normal)

    norm_w = w * THRESH["valgus_ratio"]
    l_off = valgus_offset(l_hip, l_knee, l_ankle)
    r_off = valgus_offset(r_hip, r_knee, r_ankle)
    if l_off > norm_w:
        risks.append(("warn", "左膝内扣"))
    if r_off < -norm_w:
        risks.append(("warn", "右膝内扣"))

    return risks, {
        "trunk_ang": trunk_ang,
        "knee_ang_min": knee_ang_min,
        "l_knee_ang": l_knee_ang,
        "r_knee_ang": r_knee_ang,
    }


# ============== 课堂状态识别 ==============
class StateRecognizer:
    """基于关键点帧间位移判断 站立/运动/静止"""

    def __init__(self, history_len=8, move_thresh=0.012):
        self.history = deque(maxlen=history_len)
        self.move_thresh = move_thresh

    def update(self, lm):
        if lm is None:
            return "未检测"
        feat = np.array([
            lm[L_SHO].x, lm[L_SHO].y, lm[R_SHO].x, lm[R_SHO].y,
            lm[L_WRIST].x, lm[L_WRIST].y, lm[R_WRIST].x, lm[R_WRIST].y,
            lm[L_ANKLE].x, lm[L_ANKLE].y, lm[R_ANKLE].x, lm[R_ANKLE].y,
        ])
        self.history.append(feat)
        if len(self.history) < 2:
            return "静止"
        motion = np.mean(np.abs(np.diff(np.array(self.history), axis=0)))
        if motion < self.move_thresh * 0.4:
            return "静止"
        elif motion < self.move_thresh:
            return "站立"
        return "运动"


# ============== 可视化 ==============
COLORS = {
    "danger": (0, 0, 255), "warn": (0, 165, 255), "info": (255, 255, 0),
    "safe": (0, 200, 0), "text": (255, 255, 255), "panel": (40, 40, 40),
}


def draw_skeleton(img, lm, w, h):
    """自定义骨架绘制（替代旧版 drawing_utils）"""
    pts = [(to_pixel(lm[i], w, h)) for i in range(33)]
    for a, b in POSE_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], (255, 255, 255), 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, p, 3, (0, 255, 255), -1, cv2.LINE_AA)


def draw_panel(img, risks, metrics, state, fps):
    h, w = img.shape[:2]
    panel_w, panel_h = 360, 150
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), COLORS["panel"], -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    y = 38
    cv2.putText(img, f"FPS: {fps:.1f}   状态: {state}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["text"], 1)
    y += 28
    cv2.putText(img, f"躯干前倾: {metrics.get('trunk_ang', 0):.0f} deg", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["text"], 1)
    y += 24
    cv2.putText(img, f"膝关节角(min): {metrics.get('knee_ang_min', 0):.0f} deg", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["text"], 1)
    y += 24

    if not risks:
        cv2.putText(img, "Action OK", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["safe"], 2)
    else:
        alarm_y = h - 30 - len(risks) * 30
        for i, (level, desc) in enumerate(risks):
            cv2.rectangle(img, (10, alarm_y + i * 30 - 22),
                          (w - 10, alarm_y + i * 30 + 6), COLORS[level], -1)
            cv2.putText(img, f"[{level.upper()}] {desc}", (20, alarm_y + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


# ============== 主流程 ==============
def run(input_source, output_path=None, model_path=None, no_display=False):
    model_path = ensure_model(model_path)

    cap = cv2.VideoCapture(input_source if input_source else 0)
    if not cap.isOpened():
        print(f"[错误] 无法打开视频源: {input_source or '摄像头'}")
        return

    writer = None
    if output_path:
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 25
        w_src = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h_src = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps_src, (w_src, h_src))

    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    state_rec = StateRecognizer()
    prev_t = time.time()
    fps_smooth = 0.0
    frame_idx = 0

    with vision.PoseLandmarker.create_from_options(options) as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC)) or frame_idx * 33
            result = detector.detect_for_video(mp_image, ts_ms)
            frame_idx += 1

            risks, metrics = [], {"trunk_ang": 0, "knee_ang_min": 0}
            state = "未检测"

            if result.pose_landmarks:
                lm = result.pose_landmarks[0]  # 单人
                risks, metrics = detect_risks(lm, w, h)
                state = state_rec.update(lm)
                draw_skeleton(frame, lm, w, h)

            now = time.time()
            inst_fps = 1.0 / max(now - prev_t, 1e-6)
            prev_t = now
            fps_smooth = 0.9 * fps_smooth + 0.1 * inst_fps if fps_smooth else inst_fps

            draw_panel(frame, risks, metrics, state, fps_smooth)

            if writer:
                writer.write(frame)
            if not no_display:
                cv2.imshow("BlazePose Sports Monitor (press q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[完成] 监测结束。")


def main():
    parser = argparse.ArgumentParser(description="BlazePose 校园体育课堂危险动作监测 Demo")
    parser.add_argument("--input", "-i", default="", help="视频文件路径，留空则用摄像头")
    parser.add_argument("--output", "-o", default="", help="结果视频保存路径（可选）")
    parser.add_argument("--model", "-m", default="", help="BlazePose 模型路径（可选）")
    parser.add_argument("--no-display", action="store_true", help="无显示模式，仅输出视频文件")
    args = parser.parse_args()
    run(args.input.strip() or None, args.output.strip() or None,
        args.model.strip() or None, args.no_display)


if __name__ == "__main__":
    main()
