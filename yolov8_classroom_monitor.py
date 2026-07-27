# -*- coding: utf-8 -*-
"""
校园智慧体育课堂 - YOLOv8 多人姿态监测主引擎
================================================
功能：
  1. YOLOv8-pose 多人实时检测（COCO 17 关键点）
  2. 危险动作规则库：弯腰过猛 / 膝盖过脚尖 / 屈膝过度 / 膝盖内扣
  3. 课堂状态识别：站立 / 运动 / 静止
  4. 课堂统计：到课人数、运动人数、静止人数、风险告警数
  5. 可视化：每人骨架、告警标注、统计面板
  6. 支持摄像头/视频文件，可选保存结果

用法：
  摄像头实时监测：  python yolov8_classroom_monitor.py
  指定视频文件：    python yolov8_classroom_monitor.py --input test.mp4
  保存结果视频：    python yolov8_classroom_monitor.py --input test.mp4 --output out.mp4
  无显示模式：      python yolov8_classroom_monitor.py --input test.mp4 --output out.mp4 --no-display

依赖：ultralytics opencv-python numpy
模型：yolov8n-pose.pt（首次运行自动下载）
"""

import argparse
import math
import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

# ============== COCO 17 关键点索引（YOLOv8-pose）==============
# 0:nose 1:l_eye 2:r_eye 3:l_ear 4:r_ear
# 5:l_shoulder 6:r_shoulder 7:l_elbow 8:r_elbow 9:l_wrist 10:r_wrist
# 11:l_hip 12:r_hip 13:l_knee 14:r_knee 15:l_ankle 16:r_ankle
NOSE = 0
L_SHO, R_SHO = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

# COCO 17 点骨架连接
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # 头部
    (5, 6),                                    # 肩
    (5, 7), (7, 9),                            # 左臂
    (6, 8), (8, 10),                           # 右臂
    (5, 11), (6, 12), (11, 12),                # 躯干
    (11, 13), (13, 15),                        # 左腿
    (12, 14), (14, 16),                        # 右腿
]

# ============== 危险动作阈值 ==============
THRESH = {
    "trunk_forward_warn": 55,
    "trunk_forward_danger": 75,
    "knee_over_toe_margin": 0.02,
    "squat_knee_open_min": 150,
    "valgus_ratio": 0.04,
    "keypoint_confidence": 0.3,   # 关键点置信度低于此值则跳过该点
}


# ============== 几何工具 ==============
def get_kp(keypoints, idx, w, h, conf_thresh=THRESH["keypoint_confidence"]):
    """获取关键点像素坐标，返回 (px, py, conf)"""
    kp = keypoints[idx]
    x, y, conf = float(kp[0]), float(kp[1]), float(kp[2])
    if conf < conf_thresh:
        return None
    return (int(x), int(y), conf)


def angle_3pts(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos_v = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return math.degrees(math.acos(np.clip(cos_v, -1.0, 1.0)))


def midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def trunk_forward_angle(sho_mid, hip_mid):
    dx = sho_mid[0] - hip_mid[0]
    dy = sho_mid[1] - hip_mid[1]
    return math.degrees(math.atan2(abs(dx), abs(dy) + 1e-8))


# ============== 单人危险动作检测 ==============
def detect_risks_person(keypoints, w, h):
    """对单人关键点做危险动作检测，返回 (风险列表, 指标)"""
    risks = []

    l_sho = get_kp(keypoints, L_SHO, w, h)
    r_sho = get_kp(keypoints, R_SHO, w, h)
    l_hip = get_kp(keypoints, L_HIP, w, h)
    r_hip = get_kp(keypoints, R_HIP, w, h)
    l_knee = get_kp(keypoints, L_KNEE, w, h)
    r_knee = get_kp(keypoints, R_KNEE, w, h)
    l_ankle = get_kp(keypoints, L_ANKLE, w, h)
    r_ankle = get_kp(keypoints, R_ANKLE, w, h)

    metrics = {"trunk_ang": 0, "knee_ang_min": 0}

    # 需要肩膀和髋部才能算躯干前倾
    if l_sho and r_sho and l_hip and r_hip:
        sho_mid = midpoint(l_sho[:2], r_sho[:2])
        hip_mid = midpoint(l_hip[:2], r_hip[:2])
        trunk_ang = trunk_forward_angle(sho_mid, hip_mid)
        metrics["trunk_ang"] = trunk_ang
        if trunk_ang >= THRESH["trunk_forward_danger"]:
            risks.append(("danger", "弯腰过猛"))
        elif trunk_ang >= THRESH["trunk_forward_warn"]:
            risks.append(("warn", "躯干前倾"))

    # 膝关节角与膝盖过脚尖
    knee_angs = []
    for side, hip, knee, ankle in [
        ("左", l_hip, l_knee, l_ankle),
        ("右", r_hip, r_knee, r_ankle),
    ]:
        if hip and knee and ankle:
            kang = angle_3pts(hip[:2], knee[:2], ankle[:2])
            knee_angs.append(kang)
            # 屈膝时才检查膝盖过脚尖
            if kang < 160:
                # 用归一化 x 比较（YOLOv8 输出的是像素坐标，需转归一化）
                knee_nx = knee[0] / w
                # COCO 没有 foot_index，用 ankle 的 x 近似
                ankle_nx = ankle[0] / w
                if abs(knee_nx - ankle_nx) > THRESH["knee_over_toe_margin"] * 3:
                    risks.append(("warn", f"{side}膝过脚尖"))

    if knee_angs:
        knee_ang_min = min(knee_angs)
        metrics["knee_ang_min"] = knee_ang_min
        if knee_ang_min < THRESH["squat_knee_open_min"]:
            risks.append(("warn", f"屈膝过度"))

    # 膝盖内扣
    if l_hip and l_knee and l_ankle:
        l_off = _valgus_offset(l_hip[:2], l_knee[:2], l_ankle[:2])
        if abs(l_off) > h * THRESH["valgus_ratio"]:
            risks.append(("warn", "左膝内扣"))
    if r_hip and r_knee and r_ankle:
        r_off = _valgus_offset(r_hip[:2], r_knee[:2], r_ankle[:2])
        if abs(r_off) > h * THRESH["valgus_ratio"]:
            risks.append(("warn", "右膝内扣"))

    return risks, metrics


def _valgus_offset(hip, knee, ankle):
    hip, knee, ankle = np.array(hip), np.array(knee), np.array(ankle)
    line = ankle - hip
    norm = np.linalg.norm(line) + 1e-8
    normal = np.array([-line[1], line[0]]) / norm
    return np.dot(knee - hip, normal)


# ============== 多人状态跟踪 ==============
class PersonState:
    """单人的运动状态历史"""

    def __init__(self, pid, history_len=10):
        self.pid = pid
        self.history = deque(maxlen=history_len)
        self.state = "未检测"

    def update(self, keypoints, w, h):
        # 用肩、腕、踝的中心点作为特征
        feat_pts = []
        for idx in [L_SHO, R_SHO, L_WRIST, R_WRIST, L_ANKLE, R_ANKLE]:
            kp = get_kp(keypoints, idx, w, h)
            if kp:
                feat_pts.extend([kp[0] / w, kp[1] / h])
        if len(feat_pts) < 4:
            return "未检测"
        self.history.append(feat_pts)
        if len(self.history) < 2:
            self.state = "静止"
            return self.state
        motion = np.mean(np.abs(np.diff(np.array(self.history), axis=0)))
        if motion < 0.005:
            self.state = "静止"
        elif motion < 0.015:
            self.state = "站立"
        else:
            self.state = "运动"
        return self.state


class ClassroomTracker:
    """课堂多人状态管理：基于空间位置做简单 ID 关联"""

    def __init__(self, max_persons=50):
        self.persons = {}  # pid -> PersonState
        self.next_pid = 0

    def update(self, detections, w, h):
        """
        detections: list of (keypoints, box) 
        返回: [(pid, state, risks, metrics, box), ...]
        """
        results = []
        # 简单策略：按 x 中心排序分配 ID（适用于课堂固定机位场景）
        sorted_det = sorted(detections, key=lambda d: (d[1][0] + d[1][2]) / 2)

        for i, (kpts, box) in enumerate(sorted_det):
            if i not in self.persons:
                self.persons[i] = PersonState(i)
            state = self.persons[i].update(kpts, w, h)
            risks, metrics = detect_risks_person(kpts, w, h)
            results.append((i, state, risks, metrics, box))
        return results

    def get_stats(self, results):
        moving = sum(1 for _, s, _, _, _ in results if s == "运动")
        standing = sum(1 for _, s, _, _, _ in results if s == "站立")
        still = sum(1 for _, s, _, _, _ in results if s == "静止")
        alerts = sum(len(r) for _, _, r, _, _ in results)
        return {
            "total": len(results),
            "moving": moving,
            "standing": standing,
            "still": still,
            "alerts": alerts,
        }


# ============== 可视化 ==============
COLORS = {
    "danger": (0, 0, 255),
    "warn": (0, 165, 255),
    "safe": (0, 200, 0),
    "skeleton": (0, 255, 255),
    "bone": (255, 255, 255),
    "text": (255, 255, 255),
    "panel": (40, 40, 40),
    "box": (100, 200, 255),
    "moving": (0, 200, 0),
    "standing": (200, 200, 0),
    "still": (128, 128, 128),
}


def draw_person(img, pid, state, risks, box, w, h):
    """绘制单人框、ID、状态、告警"""
    x1, y1, x2, y2 = [int(v) for v in box]
    color = COLORS.get(state, COLORS["box"])
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    label = f"#{pid} {state}"
    if risks:
        top_risk = risks[0]
        label += f" [{top_risk[1]}]"
    cv2.rectangle(img, (x1, y1 - 22), (x1 + len(label) * 12 + 10, y1), color, -1)
    cv2.putText(img, label, (x1 + 5, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["text"], 1)


def draw_keypoints(img, keypoints, w, h):
    """绘制 COCO 17 点骨架"""
    pts = {}
    for i in range(17):
        kp = get_kp(keypoints, i, w, h)
        if kp:
            pts[i] = (kp[0], kp[1])
    for a, b in SKELETON:
        if a in pts and b in pts:
            cv2.line(img, pts[a], pts[b], COLORS["bone"], 2, cv2.LINE_AA)
    for p in pts.values():
        cv2.circle(img, p, 3, COLORS["skeleton"], -1, cv2.LINE_AA)


def draw_classroom_panel(img, stats, fps):
    """左上角课堂统计面板"""
    h, w = img.shape[:2]
    panel_w, panel_h = 340, 180
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h),
                  COLORS["panel"], -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    y = 38
    cv2.putText(img, f"FPS: {fps:.1f}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["text"], 2)
    y += 30
    cv2.putText(img, f"到课人数: {stats['total']}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["text"], 1)
    y += 26
    cv2.putText(img, f"运动: {stats['moving']}  站立: {stats['standing']}  静止: {stats['still']}",
                (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["text"], 1)
    y += 26
    alert_color = COLORS["danger"] if stats["alerts"] > 0 else COLORS["safe"]
    cv2.putText(img, f"风险告警: {stats['alerts']}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, alert_color, 2)


# ============== 主流程 ==============
def run(input_source, output_path=None, no_display=False, model_name="yolov8n-pose.pt"):
    model = YOLO(model_name)
    print(f"[信息] 模型加载完成: {model_name}")

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

    tracker = ClassroomTracker()
    prev_t = time.time()
    fps_smooth = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]

        # YOLOv8-pose 推理
        results = model(frame, verbose=False, conf=0.4)

        detections = []
        if results and results[0].keypoints is not None:
            kpts_data = results[0].keypoints.data  # (N, 17, 3)
            boxes = results[0].boxes
            for i in range(len(kpts_data)):
                box = boxes[i].xyxy[0].cpu().numpy()
                kpts = kpts_data[i].cpu().numpy()
                detections.append((kpts, box))

        # 课堂跟踪与状态识别
        person_results = tracker.update(detections, w, h)
        stats = tracker.get_stats(person_results)

        # 绘制
        for pid, state, risks, metrics, box in person_results:
            draw_keypoints(frame, detections[pid][0] if pid < len(detections) else None, w, h)
            draw_person(frame, pid, state, risks, box, w, h)

        now = time.time()
        inst_fps = 1.0 / max(now - prev_t, 1e-6)
        prev_t = now
        fps_smooth = 0.9 * fps_smooth + 0.1 * inst_fps if fps_smooth else inst_fps

        draw_classroom_panel(frame, stats, fps_smooth)

        if writer:
            writer.write(frame)
        if not no_display:
            cv2.imshow("YOLOv8 Classroom Monitor (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    if not no_display:
        cv2.destroyAllWindows()
    print("[完成] 监测结束。")


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 校园体育课堂多人监测主引擎")
    parser.add_argument("--input", "-i", default="", help="视频文件路径，留空则用摄像头")
    parser.add_argument("--output", "-o", default="", help="结果视频保存路径")
    parser.add_argument("--model", "-m", default="yolov8n-pose.pt", help="YOLOv8 模型名称")
    parser.add_argument("--no-display", action="store_true", help="无显示模式")
    args = parser.parse_args()
    run(args.input.strip() or None, args.output.strip() or None,
        args.no_display, args.model)


if __name__ == "__main__":
    main()
