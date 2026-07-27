# -*- coding: utf-8 -*-
"""
校园智慧体育课堂 - 智慧体育教学服务平台（整合版）
====================================================
统一系统架构：
  1. YOLOv8-pose 多人检测主引擎（COCO 17点）—— 课堂全员监测
  2. BlazePose 精细分析辅助模块（33点+3D）—— 对告警人员自动启动深度分析
  3. 统一可视化：课堂统计面板 + 每人状态/告警 + 精细分析弹窗
  4. 课堂报表：到课人数、运动参与度、风险告警数、告警类型分布

主次关系（呼应项目计划书）：
  - 主体：智慧体育教学服务（课堂监测、安全防护、教学减负）
  - 辅助：YOLOv8 负责多人实时感知，BlazePose 负责单人风险精细确认
  逻辑：为解决体育课安全+教学问题 → 选用视觉技术辅助实现

用法：
  摄像头实时监测：  python sports_classroom_system.py
  指定视频文件：    python sports_classroom_system.py --input test.mp4
  保存结果视频：    python sports_classroom_system.py --input test.mp4 --output out.mp4
  无显示模式：      python sports_classroom_system.py --input test.mp4 --output out.mp4 --no-display
  仅YOLO不启用BlazePose：python sports_classroom_system.py --no-blazepose

依赖：ultralytics mediapipe opencv-python numpy
"""

import argparse
import math
import os
import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

# BlazePose 按需加载（避免无 mediapipe 时整体崩溃）
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    BLAZEPOSE_AVAILABLE = True
except Exception:
    BLAZEPOSE_AVAILABLE = False


# ============== YOLOv8 COCO 17 点索引 ==============
YO_NOSE = 0
YO_L_SHO, YO_R_SHO = 5, 6
YO_L_ELBOW, YO_R_ELBOW = 7, 8
YO_L_WRIST, YO_R_WRIST = 9, 10
YO_L_HIP, YO_R_HIP = 11, 12
YO_L_KNEE, YO_R_KNEE = 13, 14
YO_L_ANKLE, YO_R_ANKLE = 15, 16

YOLO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# ============== BlazePose 33 点索引（精细分析用）==============
BZ_NOSE = 0
BZ_L_SHO, BZ_R_SHO = 11, 12
BZ_L_ELBOW, BZ_R_ELBOW = 13, 14
BZ_L_WRIST, BZ_R_WRIST = 15, 16
BZ_L_HIP, BZ_R_HIP = 23, 24
BZ_L_KNEE, BZ_R_KNEE = 25, 26
BZ_L_ANKLE, BZ_R_ANKLE = 27, 28
BZ_L_HEEL, BZ_R_HEEL = 29, 30
BZ_L_FOOT, BZ_R_FOOT = 31, 32

BZ_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]

# ============== 基础阈值（已调优：大幅降低误报）==============
# 设计原则：只在真正危险时告警，正常运动/抬手/屈膝不触发
BASE_THRESH = {
    "trunk_forward_warn": 65,      # 躯干前倾警告
    "trunk_forward_danger": 80,    # 躯干前倾危险
    "knee_bent_angle": 150,        # 膝关节角小于此值才算"弯曲"
    "squat_knee_open_min": 120,    # 屈膝过度阈值
    "valgus_ratio": 0.08,          # 膝盖内扣
    "kp_confidence": 0.35,         # 关键点置信度
    "knee_over_toe_ratio": 0.08,   # 膝盖过脚尖
    "squat_back_arch_trunk": 55,   # 深蹲塌腰躯干角
    "squat_back_arch_knee": 100,   # 深蹲塌腰膝关节角
    "elbow_overextend": 185,       # 手臂过伸
    "neck_back_tilt_ratio": 0.15,  # 颈部后仰偏移
    "neck_below_shoulder": 0.05,   # 颈部低于肩膀
    "hand_raise_head_dist": 0.22,  # 举手距离
    "shoulder_tilt_ratio": 0.10,   # 懒散站姿肩高差
    "waist_hand_dist": 0.12,       # 叉腰距离
    "squat_knee_angle": 100,       # 深蹲膝关节角
    "squat_trunk_min": 25,         # 深蹲躯干前倾下限
    "squat_trunk_max": 65,         # 深蹲躯干前倾上限
    "jump_feet_ratio": 1.6,        # 开合跳脚距比
    "jump_ankle_ratio": 0.82,      # 跳跃脚踝高度比
}

# ============== 场景差异化配置 ==============
# 不同场景使用不同检测项和阈值，实现真正的区分度
SCENE_PROFILES = {
    "campus": {
        "name": "校园体育",
        # 校园：侧重课堂行为、参与度、安全告警
        "enable_classroom_behavior": True,   # 课堂行为识别（举手/蹲下/懒散等）
        "enable_exercise_action": False,     # 不侧重运动动作
        "enable_risk_detection": True,       # 安全风险检测
        "enable_blazepose_confirm": True,    # BlazePose精细确认
        "risk_sensitivity": "standard",      # 标准风险阈值
        "panel_title": "课堂监测",
    },
    "fitness": {
        "name": "健身训练",
        # 健身：侧重动作识别、姿势纠正、训练强度，风险更敏感
        "enable_classroom_behavior": False,
        "enable_exercise_action": True,      # 运动动作识别（深蹲/开合跳/高抬腿等）
        "enable_risk_detection": True,
        "enable_blazepose_confirm": True,
        "risk_sensitivity": "strict",        # 更严格的风险阈值（健身动作要求标准）
        "panel_title": "健身训练",
    },
    "rehab": {
        "name": "康复训练",
        # 康复：侧重关节活动度、温和提示，风险阈值放宽（避免过度告警）
        "enable_classroom_behavior": False,
        "enable_exercise_action": True,      # 识别康复动作
        "enable_risk_detection": True,
        "enable_blazepose_confirm": True,
        "risk_sensitivity": "gentle",        # 宽松阈值（只报严重风险）
        "panel_title": "康复训练",
    },
    "general": {
        "name": "通用模式",
        # 通用：全部检测项开启，标准阈值
        "enable_classroom_behavior": True,
        "enable_exercise_action": True,
        "enable_risk_detection": True,
        "enable_blazepose_confirm": True,
        "risk_sensitivity": "standard",
        "panel_title": "综合检测",
    },
}


def get_scene_thresh(scene="campus"):
    """根据场景返回调整后的阈值"""
    import copy
    thresh = copy.deepcopy(BASE_THRESH)
    sensitivity = SCENE_PROFILES.get(scene, SCENE_PROFILES["campus"])["risk_sensitivity"]
    if sensitivity == "strict":
        # 健身：更严格，动作不标准就告警
        thresh["trunk_forward_warn"] = 55
        thresh["trunk_forward_danger"] = 70
        thresh["squat_knee_open_min"] = 130
        thresh["valgus_ratio"] = 0.05
        thresh["knee_over_toe_ratio"] = 0.05
        thresh["elbow_overextend"] = 180
        thresh["squat_knee_angle"] = 110
    elif sensitivity == "gentle":
        # 康复：更宽松，只有严重风险才告警
        thresh["trunk_forward_warn"] = 75
        thresh["trunk_forward_danger"] = 90
        thresh["squat_knee_open_min"] = 100
        thresh["valgus_ratio"] = 0.12
        thresh["knee_over_toe_ratio"] = 0.12
        thresh["elbow_overextend"] = 190
        thresh["squat_knee_angle"] = 90
    return thresh


# 全局阈值（默认校园场景，可被动态替换）
THRESH = get_scene_thresh("campus")

# ============== 颜色 ==============
COLORS = {
    "danger": (0, 0, 255), "warn": (0, 165, 255), "safe": (0, 200, 0),
    "skeleton": (0, 255, 255), "bone": (255, 255, 255),
    "text": (255, 255, 255), "panel": (40, 40, 40),
    "moving": (0, 200, 0), "standing": (200, 200, 0), "still": (128, 128, 128),
    "blazepose": (255, 100, 255),
}


# ============== 几何工具 ==============
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


def valgus_offset(hip, knee, ankle):
    hip, knee, ankle = np.array(hip), np.array(knee), np.array(ankle)
    line = ankle - hip
    norm = np.linalg.norm(line) + 1e-8
    normal = np.array([-line[1], line[0]]) / norm
    return np.dot(knee - hip, normal)


# ============== YOLOv8 基础危险动作检测（17点）==============
def detect_risks_yolo(kpts, w, h):
    """用 YOLOv8 17 点做快速风险筛查，返回 (风险列表, 指标)"""
    risks = []

    def kp(idx):
        k = kpts[idx]
        if float(k[2]) < THRESH["kp_confidence"]:
            return None
        return (int(k[0]), int(k[1]), float(k[2]))

    nose = kp(YO_NOSE)
    l_sho, r_sho = kp(YO_L_SHO), kp(YO_R_SHO)
    l_elbow, r_elbow = kp(YO_L_ELBOW), kp(YO_R_ELBOW)
    l_wrist, r_wrist = kp(YO_L_WRIST), kp(YO_R_WRIST)
    l_hip, r_hip = kp(YO_L_HIP), kp(YO_R_HIP)
    l_knee, r_knee = kp(YO_L_KNEE), kp(YO_R_KNEE)
    l_ankle, r_ankle = kp(YO_L_ANKLE), kp(YO_R_ANKLE)

    metrics = {"trunk_ang": 0, "knee_ang_min": 0}
    knee_angs = []

    # 躯干前倾
    if l_sho and r_sho and l_hip and r_hip:
        trunk_ang = trunk_forward_angle(midpoint(l_sho[:2], r_sho[:2]),
                                         midpoint(l_hip[:2], r_hip[:2]))
        metrics["trunk_ang"] = trunk_ang
        if trunk_ang >= THRESH["trunk_forward_danger"]:
            risks.append(("danger", "弯腰过猛"))
        elif trunk_ang >= THRESH["trunk_forward_warn"]:
            risks.append(("warn", "躯干前倾"))

    # 膝关节
    for hip, knee, ankle in [(l_hip, l_knee, l_ankle), (r_hip, r_knee, r_ankle)]:
        if hip and knee and ankle:
            kang = angle_3pts(hip[:2], knee[:2], ankle[:2])
            knee_angs.append(kang)
    if knee_angs:
        knee_ang_min = min(knee_angs)
        metrics["knee_ang_min"] = knee_ang_min
        if knee_ang_min < THRESH["squat_knee_open_min"]:
            risks.append(("warn", "屈膝过度"))

    # 膝盖内扣
    valgus_flag = False
    for hip, knee, ankle in [(l_hip, l_knee, l_ankle), (r_hip, r_knee, r_ankle)]:
        if hip and knee and ankle:
            off = valgus_offset(hip[:2], knee[:2], ankle[:2])
            if abs(off) > h * THRESH["valgus_ratio"]:
                risks.append(("warn", "膝盖内扣"))
                valgus_flag = True
                break

    # ===== 新增风险检测 =====
    # 膝盖过脚尖：仅在膝关节实际弯曲（角度<160°）时检查，避免站立误报
    if knee_angs:
        for hip, knee, ankle in [(l_hip, l_knee, l_ankle), (r_hip, r_knee, r_ankle)]:
            if hip and knee and ankle:
                kang = angle_3pts(hip[:2], knee[:2], ankle[:2])
                if kang < THRESH["knee_bent_angle"]:
                    # 膝盖水平位置明显超过脚踝（向前突出）
                    if abs(knee[0] - ankle[0]) > w * THRESH["knee_over_toe_ratio"]:
                        risks.append(("warn", "膝盖过脚尖"))
                        break

    # 深蹲塌腰：躯干前倾>45° 且 膝关节角<120°（腰椎过度受力）
    if metrics["trunk_ang"] > THRESH["squat_back_arch_trunk"] and \
       metrics["knee_ang_min"] and metrics["knee_ang_min"] < THRESH["squat_back_arch_knee"]:
        risks.append(("danger", "深蹲塌腰"))

    # 手臂过伸：肘关节角度>178°（手臂过度伸展）
    for sho, elbow, wrist in [(l_sho, l_elbow, l_wrist), (r_sho, r_elbow, r_wrist)]:
        if sho and elbow and wrist:
            eang = angle_3pts(sho[:2], elbow[:2], wrist[:2])
            if eang > THRESH["elbow_overextend"]:
                risks.append(("warn", "手臂过伸"))
                break

    # 颈部后仰：鼻子相对肩膀中点位置异常（鼻子低于肩膀 或 水平偏移过大）
    if nose and l_sho and r_sho:
        sho_mid = midpoint(l_sho[:2], r_sho[:2])
        dx_neck = abs(nose[0] - sho_mid[0])
        dy_neck = nose[1] - sho_mid[1]  # 正常为负（鼻子在肩膀上方）
        if dy_neck > h * THRESH["neck_below_shoulder"] or \
           dx_neck > w * THRESH["neck_back_tilt_ratio"]:
            risks.append(("warn", "颈部后仰"))

    # 跳跃落地风险：膝盖内扣 + 屈膝过度 同时出现时升级为 danger
    has_valgus = valgus_flag
    has_knee_bent = metrics["knee_ang_min"] and \
                    metrics["knee_ang_min"] < THRESH["squat_knee_open_min"]
    if has_valgus and has_knee_bent:
        risks.append(("danger", "跳跃落地风险"))

    return risks, metrics


# ============== 课堂行为识别（YOLOv8 17点）==============
def detect_classroom_behavior(kpts, w, h):
    """基于 YOLOv8 17 点识别课堂行为，返回 [(行为类型, 描述), ...]"""
    behaviors = []

    def kp(idx):
        k = kpts[idx]
        if float(k[2]) < THRESH["kp_confidence"]:
            return None
        return (int(k[0]), int(k[1]), float(k[2]))

    nose = kp(YO_NOSE)
    l_sho, r_sho = kp(YO_L_SHO), kp(YO_R_SHO)
    l_wrist, r_wrist = kp(YO_L_WRIST), kp(YO_R_WRIST)
    l_hip, r_hip = kp(YO_L_HIP), kp(YO_R_HIP)
    l_knee, r_knee = kp(YO_L_KNEE), kp(YO_R_KNEE)
    l_ankle, r_ankle = kp(YO_L_ANKLE), kp(YO_R_ANKLE)

    if not (l_sho and r_sho):
        return behaviors

    sho_mid = midpoint(l_sho[:2], r_sho[:2])
    hip_mid = midpoint(l_hip[:2], r_hip[:2]) if l_hip and r_hip else None

    # 躯干前倾角
    trunk_ang = 0.0
    if l_hip and r_hip:
        trunk_ang = trunk_forward_angle(sho_mid, hip_mid)

    # 膝关节角最小值
    knee_angs = []
    for hip, knee, ankle in [(l_hip, l_knee, l_ankle), (r_hip, r_knee, r_ankle)]:
        if hip and knee and ankle:
            knee_angs.append(angle_3pts(hip[:2], knee[:2], ankle[:2]))
    knee_ang_min = min(knee_angs) if knee_angs else 180.0

    # 举手提问：手腕高于对应肩膀 且 手在头部附近
    for sho, wrist in [(l_sho, l_wrist), (r_sho, r_wrist)]:
        if sho and wrist and nose:
            if wrist[1] < sho[1] and \
               abs(wrist[0] - nose[0]) < w * THRESH["hand_raise_head_dist"] and \
               abs(wrist[1] - nose[1]) < h * 0.25:
                behaviors.append(("举手提问", "手举过肩且靠近头部"))
                break

    # 蹲下休息：膝关节角<100° 且 躯干接近垂直(前倾角<30°)
    if knee_ang_min < 100.0 and trunk_ang < 30.0:
        behaviors.append(("蹲下休息", "屈膝且躯干直立"))

    # 双手抱头：双手腕都高于肩膀 且 接近头部高度
    if l_wrist and r_wrist and nose:
        if l_wrist[1] < max(l_sho[1], r_sho[1]) and r_wrist[1] < max(l_sho[1], r_sho[1]):
            if abs(l_wrist[1] - nose[1]) < h * 0.18 and abs(r_wrist[1] - nose[1]) < h * 0.18:
                behaviors.append(("双手抱头", "双手高于肩且贴近头部"))

    # 叉腰：手腕在髋部附近（贴近髋部高度），且接近躯干中线
    if l_wrist and r_wrist and l_hip and r_hip:
        y_hi = min(l_sho[1], r_sho[1])
        y_lo = max(l_hip[1], r_hip[1])
        y_range = max(y_lo - y_hi, 1)
        for wrist in [l_wrist, r_wrist]:
            # 手腕须在肩-髋区间的下半部分（贴近髋部，符合叉腰特征）
            if wrist[1] > y_hi + y_range * 0.5 and wrist[1] < y_lo + y_range * 0.15 and \
               abs(wrist[0] - sho_mid[0]) < w * THRESH["waist_hand_dist"]:
                behaviors.append(("叉腰", "手腕贴近髋部且位于躯干中线"))
                break

    # 懒散站姿：肩膀左右高度差超过阈值（身体歪斜）
    if l_sho and r_sho:
        if abs(l_sho[1] - r_sho[1]) > h * THRESH["shoulder_tilt_ratio"]:
            behaviors.append(("懒散站姿", "双肩高度差过大"))

    return behaviors


# ============== 运动动作识别（YOLOv8 17点）==============
def detect_exercise_action(kpts, w, h, history=None):
    """基于当前帧姿态判断运动动作，返回动作名称字符串或 None"""
    def kp(idx):
        k = kpts[idx]
        if float(k[2]) < THRESH["kp_confidence"]:
            return None
        return (int(k[0]), int(k[1]), float(k[2]))

    nose = kp(YO_NOSE)
    l_sho, r_sho = kp(YO_L_SHO), kp(YO_R_SHO)
    l_wrist, r_wrist = kp(YO_L_WRIST), kp(YO_R_WRIST)
    l_hip, r_hip = kp(YO_L_HIP), kp(YO_R_HIP)
    l_knee, r_knee = kp(YO_L_KNEE), kp(YO_R_KNEE)
    l_ankle, r_ankle = kp(YO_L_ANKLE), kp(YO_R_ANKLE)

    if not (l_sho and r_sho and l_hip and r_hip and l_knee and r_knee and l_ankle and r_ankle):
        return None

    sho_mid = midpoint(l_sho[:2], r_sho[:2])
    hip_mid = midpoint(l_hip[:2], r_hip[:2])
    shoulder_w = abs(l_sho[0] - r_sho[0])
    feet_dist = abs(l_ankle[0] - r_ankle[0])

    trunk_ang = trunk_forward_angle(sho_mid, hip_mid)
    l_kang = angle_3pts(l_hip[:2], l_knee[:2], l_ankle[:2])
    r_kang = angle_3pts(r_hip[:2], r_knee[:2], r_ankle[:2])
    knee_ang_min = min(l_kang, r_kang)

    # 深蹲：膝关节角<110° 且 躯干前倾角30°-60°
    if knee_ang_min < THRESH["squat_knee_angle"] and \
       THRESH["squat_trunk_min"] <= trunk_ang <= THRESH["squat_trunk_max"]:
        return "深蹲"

    # 开合跳：双脚分开距离 > 肩宽1.5倍 且 双手腕高于肩膀
    if shoulder_w > 0 and feet_dist > shoulder_w * THRESH["jump_feet_ratio"]:
        if l_wrist and r_wrist and l_wrist[1] < sho_mid[1] and r_wrist[1] < sho_mid[1]:
            return "开合跳"

    # 高抬腿：单侧膝盖y坐标 < 髋部y坐标（膝盖抬到髋部以上）
    if l_knee[1] < hip_mid[1] or r_knee[1] < hip_mid[1]:
        return "高抬腿"

    # 跳跃：双脚踝y坐标同时明显高于髋部中点（双脚离地特征）
    if l_ankle[1] < hip_mid[1] * THRESH["jump_ankle_ratio"] and \
       r_ankle[1] < hip_mid[1] * THRESH["jump_ankle_ratio"]:
        return "跳跃"

    # 拉伸：单手高举过头顶(手腕y<鼻子y) 且 另一只手自然下垂（非深蹲非跳跃）
    if nose and l_wrist and r_wrist:
        l_up = l_wrist[1] < nose[1]
        r_up = r_wrist[1] < nose[1]
        # 一只手举起过头顶，另一只手低于髋部（下垂）
        if (l_up and not r_up and r_wrist[1] > hip_mid[1]) or \
           (r_up and not l_up and l_wrist[1] > hip_mid[1]):
            return "拉伸"

    return None


# ============== BlazePose 精细分析（33点+3D）==============
class BlazePoseAnalyzer:
    """对单人裁剪区域做 BlazePose 精细分析，确认 YOLOv8 的告警"""

    def __init__(self, model_path=None):
        if not BLAZEPOSE_AVAILABLE:
            self.available = False
            return
        self.model_path = self._ensure_model(model_path)
        self.detector = None
        self.available = True

    @staticmethod
    def _ensure_model(model_path):
        if model_path and os.path.exists(model_path):
            return model_path
        default = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "models", "pose_landmarker_lite.task")
        if os.path.exists(default):
            return default
        import urllib.request
        os.makedirs(os.path.dirname(default), exist_ok=True)
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
        urllib.request.urlretrieve(url, default)
        return default

    def init(self):
        if not self.available:
            return False
        base = mp_python.BaseOptions(model_asset_path=self.model_path)
        opts = vision.PoseLandmarkerOptions(base_options=base,
                                            running_mode=vision.RunningMode.IMAGE)
        self.detector = vision.PoseLandmarker.create_from_options(opts)
        return True

    def analyze(self, person_img):
        """对单人 BGR 图像做精细分析，返回 (33点列表, 精细风险, 指标)"""
        if not self.detector:
            return None, [], {}
        rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)
        if not result.pose_landmarks:
            return None, [], {}
        lm = result.pose_landmarks[0]
        h, w = person_img.shape[:2]
        risks, metrics = self._detect_fine(lm, w, h)
        return lm, risks, metrics

    @staticmethod
    def _detect_fine(lm, w, h):
        """BlazePose 33 点精细风险检测"""
        risks = []

        def px(i):
            return (int(lm[i].x * w), int(lm[i].y * h))

        nose = px(BZ_NOSE)
        l_sho, r_sho = px(BZ_L_SHO), px(BZ_R_SHO)
        l_elbow, r_elbow = px(BZ_L_ELBOW), px(BZ_R_ELBOW)
        l_wrist, r_wrist = px(BZ_L_WRIST), px(BZ_R_WRIST)
        l_hip, r_hip = px(BZ_L_HIP), px(BZ_R_HIP)
        l_knee, r_knee = px(BZ_L_KNEE), px(BZ_R_KNEE)
        l_ankle, r_ankle = px(BZ_L_ANKLE), px(BZ_R_ANKLE)
        l_foot, r_foot = px(BZ_L_FOOT), px(BZ_R_FOOT)

        trunk_ang = trunk_forward_angle(midpoint(l_sho, r_sho), midpoint(l_hip, r_hip))
        if trunk_ang >= THRESH["trunk_forward_danger"]:
            risks.append(("danger", f"弯腰过猛({trunk_ang:.0f}°)"))
        elif trunk_ang >= THRESH["trunk_forward_warn"]:
            risks.append(("warn", f"躯干前倾({trunk_ang:.0f}°)"))

        l_kang = angle_3pts(l_hip, l_knee, l_ankle)
        r_kang = angle_3pts(r_hip, r_knee, r_ankle)
        kang_min = min(l_kang, r_kang)
        if kang_min < THRESH["squat_knee_open_min"]:
            risks.append(("warn", f"屈膝过度({kang_min:.0f}°)"))

        l_off = valgus_offset(l_hip, l_knee, l_ankle)
        if abs(l_off) > h * THRESH["valgus_ratio"]:
            risks.append(("warn", "左膝内扣"))
        r_off = valgus_offset(r_hip, r_knee, r_ankle)
        if abs(r_off) > h * THRESH["valgus_ratio"]:
            risks.append(("warn", "右膝内扣"))

        # ===== 新增精细风险检测 =====
        # 膝盖过脚尖：使用脚尖点(foot)，膝关节弯曲时检查膝盖x是否超过脚尖x
        for knee, foot, kang in [(l_knee, l_foot, l_kang), (r_knee, r_foot, r_kang)]:
            if kang < THRESH["knee_bent_angle"]:
                if abs(knee[0] - foot[0]) > w * THRESH["knee_over_toe_ratio"]:
                    risks.append(("warn", f"膝盖过脚尖({abs(knee[0]-foot[0])/w*100:.0f}%)"))
                    break

        # 深蹲塌腰：躯干前倾>45° 且 膝关节角<120°
        if trunk_ang > THRESH["squat_back_arch_trunk"] and \
           kang_min < THRESH["squat_back_arch_knee"]:
            risks.append(("danger", f"深蹲塌腰(前倾{trunk_ang:.0f}°/膝{kang_min:.0f}°)"))

        # 手臂过伸：肘关节角度>178°
        for sho, elbow, wrist in [(l_sho, l_elbow, l_wrist), (r_sho, r_elbow, r_wrist)]:
            eang = angle_3pts(sho, elbow, wrist)
            if eang > THRESH["elbow_overextend"]:
                risks.append(("warn", f"手臂过伸({eang:.0f}°)"))
                break

        # 颈部后仰：鼻子相对肩膀中点位置异常
        sho_mid = midpoint(l_sho, r_sho)
        dx_neck = abs(nose[0] - sho_mid[0])
        dy_neck = nose[1] - sho_mid[1]
        if dy_neck > h * THRESH["neck_below_shoulder"] or \
           dx_neck > w * THRESH["neck_back_tilt_ratio"]:
            risks.append(("warn", "颈部后仰"))

        # 跳跃落地风险：膝盖内扣 + 屈膝过度 同时出现升级为 danger
        has_valgus = abs(l_off) > h * THRESH["valgus_ratio"] or \
                     abs(r_off) > h * THRESH["valgus_ratio"]
        has_knee_bent = kang_min < THRESH["squat_knee_open_min"]
        if has_valgus and has_knee_bent:
            risks.append(("danger", "跳跃落地风险"))

        return risks, {"trunk_ang": trunk_ang, "knee_ang_min": kang_min}

    def close(self):
        if self.detector:
            self.detector.close()
            self.detector = None


# ============== 课堂状态跟踪 ==============
class PersonTracker:
    def __init__(self, pid):
        self.pid = pid
        self.history = deque(maxlen=10)
        self.state = "未检测"
        # 记录最近动作与行为（时序平滑用）
        self.action_history = deque(maxlen=15)
        self.behavior_history = deque(maxlen=15)

    def _is_standing_posture(self, kpts, h):
        """姿态判定：检测人是否处于直立站立姿态（非坐/躺）。
        通过髋-膝-踝的y坐标垂直排列关系判断。
        返回 True=站立姿态, False=坐/蹲/躺/无法判断
        """
        def kp_y(idx):
            k = kpts[idx]
            if float(k[2]) >= THRESH["kp_confidence"]:
                return float(k[1]) / h
            return None

        l_hip_y, r_hip_y = kp_y(YO_L_HIP), kp_y(YO_R_HIP)
        l_knee_y, r_knee_y = kp_y(YO_L_KNEE), kp_y(YO_R_KNEE)
        l_ankle_y, r_ankle_y = kp_y(YO_L_ANKLE), kp_y(YO_R_ANKLE)

        # 至少一侧的髋-膝-踝都检测到
        for hip_y, knee_y, ankle_y in [(l_hip_y, l_knee_y, l_ankle_y),
                                        (r_hip_y, r_knee_y, r_ankle_y)]:
            if hip_y is not None and knee_y is not None and ankle_y is not None:
                # 直立：髋在膝上方，膝在踝上方（y越小越靠上）
                # 且各段间距足够（不是蹲下/坐下导致的紧凑姿态）
                hip_knee_dist = knee_y - hip_y
                knee_ankle_dist = ankle_y - knee_y
                if hip_y < knee_y < ankle_y and \
                   hip_knee_dist > 0.04 and knee_ankle_dist > 0.04:
                    return True
        return False

    def update(self, kpts, w, h, scene_profile=None):
        # 固定长度为12（6个关键点 × 2坐标），未检测到的用0填充，保证数组形状一致
        track_idxs = [YO_L_SHO, YO_R_SHO, YO_L_WRIST, YO_R_WRIST,
                      YO_L_ANKLE, YO_R_ANKLE]
        feats = []
        valid_cnt = 0
        for idx in track_idxs:
            k = kpts[idx]
            if float(k[2]) >= THRESH["kp_confidence"]:
                feats.extend([float(k[0]) / w, float(k[1]) / h])
                valid_cnt += 1
            else:
                feats.extend([0.0, 0.0])
        if valid_cnt < 2:
            return "未检测"
        self.history.append(feats)

        # 姿态判定：是否直立站立（用于区分"站立"与"静止"）
        standing = self._is_standing_posture(kpts, h)

        if len(self.history) < 2:
            # 首帧：有直立姿态默认"站立"，否则"静止"
            self.state = "站立" if standing else "静止"
        else:
            try:
                motion = np.mean(np.abs(np.diff(np.array(self.history, dtype=float), axis=0)))
            except Exception:
                motion = 0.0
            # 三级判定（结合姿态+运动）：
            # 1. 运动幅度大 → "运动"
            # 2. 有一定运动 或 直立站立姿态 → "站立"
            # 3. 不动且非直立（蹲/坐/躺） → "静止"
            if motion >= 0.012:
                self.state = "运动"
            elif motion >= 0.003:
                self.state = "站立"
            elif standing:
                self.state = "站立"   # 直立但几乎不动 → 仍算站立（在听课/准备状态）
            else:
                self.state = "静止"   # 非直立且不动 → 真正的静止（坐/蹲/躺）

        # ===== 场景差异化：按需调用检测函数，避免不必要的计算 =====
        enable_action = scene_profile is None or scene_profile.get("enable_exercise_action", True)
        enable_behavior = scene_profile is None or scene_profile.get("enable_classroom_behavior", True)

        if enable_action:
            action = detect_exercise_action(kpts, w, h, self.history)
        else:
            action = None
        self.action_history.append(action)

        if enable_behavior:
            behaviors = detect_classroom_behavior(kpts, w, h)
            if behaviors:
                self.behavior_history.append(behaviors[0][0])
            else:
                self.behavior_history.append(None)
        else:
            self.behavior_history.append(None)

        return self.state

    def get_current_action(self):
        """返回最近5帧中出现最多的动作（时序平滑，减少抖动）"""
        recent = list(self.action_history)[-5:]
        valid = [a for a in recent if a is not None]
        if not valid:
            return None
        from collections import Counter
        return Counter(valid).most_common(1)[0][0]

    def get_current_behavior(self):
        """返回最近5帧中出现最多的行为（时序平滑，减少抖动）"""
        recent = list(self.behavior_history)[-5:]
        valid = [b for b in recent if b is not None]
        if not valid:
            return None
        from collections import Counter
        return Counter(valid).most_common(1)[0][0]


# ============== 可视化 ==============
def draw_yolo_skeleton(img, kpts, w, h):
    pts = {}
    for i in range(17):
        k = kpts[i]
        if float(k[2]) >= THRESH["kp_confidence"]:
            pts[i] = (int(k[0]), int(k[1]))
    for a, b in YOLO_SKELETON:
        if a in pts and b in pts:
            cv2.line(img, pts[a], pts[b], COLORS["bone"], 2, cv2.LINE_AA)
    for p in pts.values():
        cv2.circle(img, p, 3, COLORS["skeleton"], -1, cv2.LINE_AA)


def draw_blazepose_skeleton(img, lm, x_off, y_off, scale_w, scale_h):
    """在裁剪区域坐标上绘制 BlazePose 33 点骨架"""
    pts = [(int(lm[i].x * scale_w) + x_off, int(lm[i].y * scale_h) + y_off)
           for i in range(33)]
    for a, b in BZ_SKELETON:
        cv2.line(img, pts[a], pts[b], COLORS["blazepose"], 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, p, 3, (255, 255, 0), -1, cv2.LINE_AA)


def draw_person_box(img, pid, state, risks, box, fine_risks=None,
                    action=None, behavior=None):
    x1, y1, x2, y2 = [int(v) for v in box]
    color = COLORS.get(state, COLORS["still"])
    # 有精细分析则用紫色边框
    if fine_risks is not None:
        color = COLORS["blazepose"]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    label = f"#{pid} {state}"
    all_risks = risks + (fine_risks or [])
    # [] 优先显示动作，无动作则显示首要风险
    if action:
        label += f" [{action}]"
    elif all_risks:
        label += f" [{all_risks[0][1]}]"
    # () 显示当前课堂行为
    if behavior:
        label += f" ({behavior})"
    cv2.rectangle(img, (x1, y1 - 22), (x1 + len(label) * 12 + 10, y1), color, -1)
    cv2.putText(img, label, (x1 + 5, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["text"], 1)


def draw_panel(img, stats, fps, blazepose_on, scene="campus"):
    h, w = img.shape[:2]
    profile = SCENE_PROFILES.get(scene, SCENE_PROFILES["campus"])
    title = profile["panel_title"]

    # 场景配色（面板左侧色条）
    scene_colors = {
        "campus": (79, 172, 254),    # 蓝色
        "fitness": (255, 159, 67),   # 橙色
        "rehab": (0, 216, 160),      # 绿色
        "general": (168, 85, 247),   # 紫色
    }
    accent = scene_colors.get(scene, scene_colors["campus"])

    # ===== 场景专属面板尺寸 =====
    panel_w = 380
    panel_h = 270
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h),
                  COLORS["panel"], -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    # 左侧色条（场景标识）
    cv2.rectangle(img, (10, 10), (16, 10 + panel_h), accent, -1)

    y = 38
    cv2.putText(img, f"[{title}] FPS:{fps:.1f} BP:{'ON' if blazepose_on else 'OFF'}",
                (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["text"], 1)
    y += 28

    cur_actions = stats.get("cur_actions", [])
    cur_behaviors = stats.get("cur_behaviors", [])
    scene_metrics = stats.get("scene_metrics", {})

    # ===== 场景差异化面板内容 =====
    if scene == "campus":
        # —— 校园体育：课堂行为 + 参与度 + 安全 ——
        cv2.putText(img, f"到课人数: {stats['total']}", (24, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["text"], 1)
        y += 26
        cv2.putText(img, f"运动:{stats['moving']} 站立:{stats['standing']} 静止:{stats['still']}",
                    (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["text"], 1)
        y += 24
        cv2.putText(img, f"课堂参与度: {stats['participation']:.0%}", (24, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["moving"], 1)
        y += 24
        alert_color = COLORS["danger"] if stats["alerts"] > 0 else COLORS["safe"]
        cv2.putText(img, f"安全告警: {stats['alerts']}", (24, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, alert_color, 2)
        y += 26
        if stats["alerts"] > 0:
            cv2.putText(img, f"精细确认: {stats['confirmed']}", (24, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["blazepose"], 1)
            y += 24
        cv2.putText(img, f"课堂行为: {','.join(cur_behaviors) if cur_behaviors else '正常'}",
                    (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["standing"], 1)

    elif scene == "fitness":
        # —— 健身训练：动作识别 + 姿势纠正 + 严格风险 ——
        cv2.putText(img, f"训练人数: {stats['total']}", (24, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["text"], 1)
        y += 26
        cv2.putText(img, f"活跃: {stats['moving']} 休息: {stats['still']}",
                    (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["text"], 1)
        y += 24
        cv2.putText(img, f"训练活跃度: {stats['participation']:.0%}", (24, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["moving"], 1)
        y += 24
        cv2.putText(img, f"训练动作: {','.join(cur_actions) if cur_actions else '待识别'}",
                    (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["moving"], 1)
        y += 24
        # 姿势纠正指标
        form_list = scene_metrics.get("form", [])
        if form_list:
            f0 = form_list[0]
            cv2.putText(img, f"躯干角:{f0['trunk_ang']:.0f}° 膝角:{f0['knee_ang_min']:.0f}°",
                        (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["skeleton"], 1)
            y += 24
        alert_color = COLORS["danger"] if stats["alerts"] > 0 else COLORS["safe"]
        cv2.putText(img, f"姿势风险(严格): {stats['alerts']}", (24, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, alert_color, 2)

    elif scene == "rehab":
        # —— 康复训练：关节活动度 + 温和提示 ——
        cv2.putText(img, f"训练人数: {stats['total']}", (24, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["text"], 1)
        y += 26
        cv2.putText(img, f"活动:{stats['moving']} 静养:{stats['still']}",
                    (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["text"], 1)
        y += 24
        cv2.putText(img, f"活动完成度: {stats['participation']:.0%}", (24, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["safe"], 1)
        y += 24
        # 关节活动度 (ROM)
        rom_list = scene_metrics.get("rom", [])
        if rom_list:
            r0 = rom_list[0]
            cv2.putText(img, f"关节ROM: 躯干{r0['trunk_ang']:.0f}° 膝{r0['knee_ang_min']:.0f}°",
                        (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["blazepose"], 1)
            y += 24
        cv2.putText(img, f"康复动作: {','.join(cur_actions) if cur_actions else '无'}",
                    (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["moving"], 1)
        y += 24
        alert_color = COLORS["warn"] if stats["alerts"] > 0 else COLORS["safe"]
        cv2.putText(img, f"温和提示: {stats['alerts']}", (24, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, alert_color, 2)

    else:
        # —— 通用模式：全部指标 ——
        cv2.putText(img, f"检测人数: {stats['total']}", (24, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["text"], 1)
        y += 26
        cv2.putText(img, f"运动: {stats['moving']}  站立: {stats['standing']}  静止: {stats['still']}",
                    (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["text"], 1)
        y += 24
        cv2.putText(img, f"参与度: {stats['participation']:.0%}", (24, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["text"], 1)
        y += 24
        alert_color = COLORS["danger"] if stats["alerts"] > 0 else COLORS["safe"]
        cv2.putText(img, f"风险告警: {stats['alerts']}", (24, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, alert_color, 2)
        y += 26
        if stats["alerts"] > 0:
            cv2.putText(img, f"精细确认: {stats['confirmed']}", (24, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["blazepose"], 1)
            y += 24
        cv2.putText(img, f"动作: {','.join(cur_actions) if cur_actions else '无'}",
                    (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["moving"], 1)
        y += 22
        cv2.putText(img, f"行为: {','.join(cur_behaviors) if cur_behaviors else '无'}",
                    (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["standing"], 1)


# ============== 主系统 ==============
class SportsClassroomSystem:
    def __init__(self, use_blazepose=True, yolo_model="yolov8n-pose.pt", scene="campus"):
        self.yolo = YOLO(yolo_model)
        self.use_blazepose = use_blazepose and BLAZEPOSE_AVAILABLE
        self.blazepose = None
        if self.use_blazepose:
            self.blazepose = BlazePoseAnalyzer()
            self.use_blazepose = self.blazepose.init()
            if self.use_blazepose:
                print("[系统] BlazePose 精细分析模块已就绪")
            else:
                print("[系统] BlazePose 初始化失败，仅使用 YOLOv8")
        elif use_blazepose and not BLAZEPOSE_AVAILABLE:
            print("[系统] 未安装 mediapipe，仅使用 YOLOv8")

        self.persons = {}
        # ===== 场景配置 =====
        self.scene = scene
        self.set_scene(scene)
        print(f"[系统] YOLOv8 主引擎已就绪 ({yolo_model})")
        print(f"[系统] BlazePose 辅助: {'启用' if self.use_blazepose else '关闭'}")
        print(f"[系统] 场景模式: {self.scene_profile['name']} (敏感度: {self.scene_profile['risk_sensitivity']})")

    def set_scene(self, scene):
        """切换场景模式，动态调整检测项和阈值"""
        global THRESH
        self.scene = scene
        self.scene_profile = SCENE_PROFILES.get(scene, SCENE_PROFILES["campus"])
        THRESH = get_scene_thresh(scene)  # 更新全局阈值
        print(f"[系统] 切换场景 → {self.scene_profile['name']}")

    def process_frame(self, frame):
        """处理单帧，返回标注后的帧 + 统计（根据场景配置差异化检测）"""
        h, w = frame.shape[:2]
        results = self.yolo(frame, verbose=False, conf=0.4)

        detections = []
        if results and results[0].keypoints is not None:
            kpts_data = results[0].keypoints.data
            boxes = results[0].boxes
            for i in range(len(kpts_data)):
                box = boxes[i].xyxy[0].cpu().numpy()
                kpts = kpts_data[i].cpu().numpy()
                detections.append((kpts, box))

        # 按 x 中心排序分配 ID
        detections.sort(key=lambda d: (d[1][0] + d[1][2]) / 2)

        profile = self.scene_profile
        stats = {"total": 0, "moving": 0, "standing": 0, "still": 0,
                 "alerts": 0, "confirmed": 0,
                 "actions": {}, "behaviors": {},
                 "cur_actions": [], "cur_behaviors": [],
                 "scene": self.scene,
                 "scene_metrics": {}}  # 场景专属指标

        for i, (kpts, box) in enumerate(detections):
            if i not in self.persons:
                self.persons[i] = PersonTracker(i)
            tracker = self.persons[i]
            state = tracker.update(kpts, w, h, profile)

            # ===== 场景差异化：风险检测 =====
            risks = []
            if profile["enable_risk_detection"]:
                risks, metrics = detect_risks_yolo(kpts, w, h)
                # 康复模式：额外记录关节活动度(ROM)
                if self.scene == "rehab" and metrics:
                    stats["scene_metrics"].setdefault("rom", []).append({
                        "pid": i,
                        "trunk_ang": round(metrics.get("trunk_ang", 0), 1),
                        "knee_ang_min": round(metrics.get("knee_ang_min", 0), 1),
                    })
                # 健身模式：记录姿势指标用于纠正
                if self.scene == "fitness" and metrics:
                    stats["scene_metrics"].setdefault("form", []).append({
                        "pid": i,
                        "trunk_ang": round(metrics.get("trunk_ang", 0), 1),
                        "knee_ang_min": round(metrics.get("knee_ang_min", 0), 1),
                    })

            # ===== 场景差异化：课堂行为（仅校园/通用）=====
            behaviors = []
            cur_behavior = None
            if profile["enable_classroom_behavior"]:
                behaviors = detect_classroom_behavior(kpts, w, h)
                cur_behavior = tracker.get_current_behavior()

            # ===== 场景差异化：运动动作（仅健身/康复/通用）=====
            cur_action = None
            if profile["enable_exercise_action"]:
                cur_action = tracker.get_current_action()

            # 绘制 YOLOv8 骨架
            draw_yolo_skeleton(frame, kpts, w, h)

            # 有风险且启用 BlazePose → 精细确认
            fine_risks = None
            if risks and self.use_blazepose and profile["enable_blazepose_confirm"]:
                x1, y1, x2, y2 = [int(v) for v in box]
                pad = 20
                crop = frame[max(0, y1 - pad):min(h, y2 + pad),
                             max(0, x1 - pad):min(w, x2 + pad)]
                if crop.size > 0:
                    lm, fine_risks, fine_metrics = self.blazepose.analyze(crop)
                    if lm is not None:
                        draw_blazepose_skeleton(
                            frame, lm,
                            max(0, x1 - pad), max(0, y1 - pad),
                            crop.shape[1], crop.shape[0])
                        if fine_risks:
                            stats["confirmed"] += 1
                            risks = fine_risks

            draw_person_box(frame, i, state, risks, box, fine_risks,
                            cur_action, cur_behavior)

            # 统计
            stats["total"] += 1
            if state == "运动":
                stats["moving"] += 1
            elif state == "站立":
                stats["standing"] += 1
            else:
                stats["still"] += 1
            stats["alerts"] += len(risks)

            # 动作统计（仅启用了运动动作识别的场景）
            if cur_action:
                stats["actions"][cur_action] = stats["actions"].get(cur_action, 0) + 1
                if cur_action not in stats["cur_actions"]:
                    stats["cur_actions"].append(cur_action)
            # 行为统计（仅启用了课堂行为识别的场景）
            for btype, _ in behaviors:
                stats["behaviors"][btype] = stats["behaviors"].get(btype, 0) + 1
                if btype not in stats["cur_behaviors"]:
                    stats["cur_behaviors"].append(btype)

        # 参与度 = (运动+站立) / 总人数
        if stats["total"] > 0:
            stats["participation"] = (stats["moving"] + stats["standing"]) / stats["total"]
        else:
            stats["participation"] = 0.0

        return frame, stats

    def close(self):
        if self.blazepose:
            self.blazepose.close()


def run(input_source, output_path=None, no_display=False,
        use_blazepose=True, yolo_model="yolov8n-pose.pt", scene="campus"):
    system = SportsClassroomSystem(use_blazepose=use_blazepose,
                                   yolo_model=yolo_model, scene=scene)

    cap = cv2.VideoCapture(input_source if input_source else 0)
    if not cap.isOpened():
        print(f"[错误] 无法打开视频源: {input_source or '摄像头'}")
        system.close()
        return

    writer = None
    if output_path:
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 25
        w_src = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h_src = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        # 优先使用 H.264 编码（浏览器可播放），失败则回退 mp4v
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(output_path, fourcc, fps_src, (w_src, h_src))
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"H264")
            writer = cv2.VideoWriter(output_path, fourcc, fps_src, (w_src, h_src))
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps_src, (w_src, h_src))

    prev_t = time.time()
    fps_smooth = 0.0

    # ===== 统计累积 =====
    total_frames = 0
    sum_total = 0          # 累计检测到的人数（每帧求和）
    sum_moving = 0
    sum_standing = 0
    sum_still = 0
    sum_alerts = 0
    sum_confirmed = 0
    max_total = 0          # 峰值人数
    risk_type_count = {}   # 风险类型计数
    action_count = {}      # 运动动作累计计数
    behavior_count = {}    # 课堂行为累计计数
    key_frames = []        # 关键帧（有人/有告警）
    key_frame_interval = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 100) // 6)  # 均匀抽6帧

    try:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame, stats = system.process_frame(frame)

            now = time.time()
            inst_fps = 1.0 / max(now - prev_t, 1e-6)
            prev_t = now
            fps_smooth = 0.9 * fps_smooth + 0.1 * inst_fps if fps_smooth else inst_fps

            draw_panel(frame, stats, fps_smooth, system.use_blazepose, system.scene)

            # 累积统计
            total_frames += 1
            sum_total += stats["total"]
            sum_moving += stats["moving"]
            sum_standing += stats["standing"]
            sum_still += stats["still"]
            sum_alerts += stats["alerts"]
            sum_confirmed += stats["confirmed"]
            if stats["total"] > max_total:
                max_total = stats["total"]

            # 累积动作与行为计数
            for k, v in stats["actions"].items():
                action_count[k] = action_count.get(k, 0) + v
            for k, v in stats["behaviors"].items():
                behavior_count[k] = behavior_count.get(k, 0) + v

            # 关键帧保存（均匀抽样 + 有告警的帧）
            if frame_idx % key_frame_interval == 0 or (stats["alerts"] > 0 and len(key_frames) < 10):
                if len(key_frames) < 8:
                    key_frames.append((frame_idx, frame.copy(), stats.copy()))

            if writer:
                writer.write(frame)
            if not no_display:
                cv2.imshow("Sports Classroom System (press q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_idx += 1
    finally:
        cap.release()
        if writer:
            writer.release()
        if not no_display:
            cv2.destroyAllWindows()
        system.close()

    # ===== 汇总报表 =====
    avg_total = sum_total / max(total_frames, 1)
    avg_participation = (sum_moving + sum_standing) / max(sum_total, 1)
    print("\n" + "=" * 56)
    print("         校园智慧体育课堂 - 监测汇总报表")
    print("=" * 56)
    print(f"  视频总帧数:        {total_frames}")
    print(f"  平均处理帧率:      {fps_smooth:.1f} FPS")
    print(f"  课堂峰值人数:      {max_total} 人")
    print(f"  平均检测人数:      {avg_total:.1f} 人/帧")
    print("-" * 56)
    print(f"  运动状态累计:      运动 {sum_moving} 帧 | 站立 {sum_standing} 帧 | 静止 {sum_still} 帧")
    print(f"  平均参与度:        {avg_participation*100:.1f}%")
    print("-" * 56)
    print(f"  YOLOv8 初筛告警:   {sum_alerts} 次")
    print(f"  BlazePose 确认:    {sum_confirmed} 次")
    print(f"  BlazePose 辅助:    {'启用' if system.use_blazepose else '关闭'}")
    print("-" * 56)
    print("  运动动作识别分布:")
    if action_count:
        for k, v in sorted(action_count.items(), key=lambda x: -x[1]):
            print(f"    {k}: {v} 人次")
    else:
        print("    无")
    print("-" * 56)
    print("  课堂行为识别分布:")
    if behavior_count:
        for k, v in sorted(behavior_count.items(), key=lambda x: -x[1]):
            print(f"    {k}: {v} 人次")
    else:
        print("    无")
    print("=" * 56)

    # 保存关键帧
    if key_frames:
        import os
        out_dir = os.path.dirname(output_path) if output_path else "."
        kf_dir = os.path.join(out_dir, "key_frames")
        os.makedirs(kf_dir, exist_ok=True)
        for idx, (fidx, kframe, kstats) in enumerate(key_frames):
            kf_path = os.path.join(kf_dir, f"key_frame_{idx+1}_f{fidx}.jpg")
            cv2.imwrite(kf_path, kframe)
        print(f"[关键帧] 已保存 {len(key_frames)} 张到: {kf_dir}")
    print("[完成] 系统运行结束。")


def main():
    parser = argparse.ArgumentParser(description="校园智慧体育课堂统一系统")
    parser.add_argument("--input", "-i", default="", help="视频文件路径")
    parser.add_argument("--output", "-o", default="", help="结果视频保存路径")
    parser.add_argument("--model", "-m", default="yolov8n-pose.pt", help="YOLOv8 模型")
    parser.add_argument("--no-display", action="store_true", help="无显示模式")
    parser.add_argument("--no-blazepose", action="store_true", help="禁用 BlazePose 精细分析")
    args = parser.parse_args()
    run(args.input.strip() or None, args.output.strip() or None,
        args.no_display, not args.no_blazepose, args.model)


if __name__ == "__main__":
    main()
