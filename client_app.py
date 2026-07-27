# -*- coding: utf-8 -*-
"""
智慧体育课堂监测系统 - 桌面客户端 v2.0
======================================
全面升级版：多场景模式 · 拖拽导入 · 现代化UI

功能模块：
  1. 数据仪表盘 - 总览与快速入口
  2. 实时摄像头监测
  3. 视频文件分析（支持拖拽）
  4. 课堂统计面板
  5. 历史记录管理

场景模式：
  - 校园体育：课堂质量监测 · 学生安全防护
  - 健身训练：动作标准分析 · 训练效果评估
  - 康复训练：活动范围监测 · 恢复进度追踪
  - 通用模式：全功能检测 · 多场景适配

技术栈：PyQt5 + YOLOv8 + BlazePose
"""

import sys
import os
import json
import time
import sqlite3
from datetime import datetime
from collections import Counter

# ===== 关键：必须在 cv2 / PyQt5 之前导入 torch =====
import torch  # noqa: F401
from ultralytics import YOLO  # noqa: F401

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QStackedWidget, QListWidget, QListWidgetItem,
    QFileDialog, QProgressBar, QTableWidget, QTableWidgetItem, QHeaderView,
    QFrame, QScrollArea, QGroupBox, QComboBox, QMessageBox, QSplitter,
    QGridLayout, QSizePolicy, QSpinBox, QSpacerItem, QGraphicsDropShadowEffect
)
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QSize, QPropertyAnimation,
    QEasingCurve, QPointF, pyqtProperty
)
from PyQt5.QtGui import (
    QFont, QPixmap, QImage, QColor, QPalette, QIcon, QLinearGradient,
    QBrush, QGradient, QPainter, QPen, QPainterPath, QFontDatabase
)

# ===== 检测引擎按需加载 =====
ENGINE_AVAILABLE = None
ENGINE_ERROR = ""

def load_engine():
    global ENGINE_AVAILABLE, ENGINE_ERROR
    try:
        import importlib
        mod = importlib.import_module("sports_classroom_system")
        globals()["SportsClassroomSystem"] = mod.SportsClassroomSystem
        globals()["draw_panel"] = mod.draw_panel
        ENGINE_AVAILABLE = True
        ENGINE_ERROR = ""
        return True, ""
    except Exception as e:
        ENGINE_AVAILABLE = False
        ENGINE_ERROR = str(e)
        return False, str(e)

SportsClassroomSystem = None
draw_panel = None


# ================================================================
# 场景模式配置
# ================================================================
SCENE_CONFIG = {
    "campus": {
        "name": "校园体育",
        "icon": "🎓",
        "color": "#4facfe",
        "color2": "#00f2fe",
        "gradient": "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4facfe, stop:1 #00f2fe)",
        "desc": "课堂质量监测 · 学生安全防护",
        "features": ["课堂参与度", "运动安全告警", "行为识别", "到课统计"],
        # 场景特定文案
        "camera_title": "校园课堂 · 实时监测",
        "camera_desc": "实时检测课堂中学生姿态、运动安全风险与课堂行为",
        "video_title": "校园课堂 · 视频分析",
        "video_desc": "分析课堂录像，评估学生参与度、运动安全与课堂行为",
        "state_label": "课堂状态",
        "count_label": "到课人数",
        "part_label": "课堂参与度",
        "stats_title": "课堂统计面板",
        "stats_desc": "汇总历史监测数据 · 展示课堂质量与安全趋势",
        "action_title": "运动动作识别统计",
        "behavior_title": "课堂行为识别统计",
    },
    "fitness": {
        "name": "健身训练",
        "icon": "💪",
        "color": "#ff9f43",
        "color2": "#ff7847",
        "gradient": "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ff9f43, stop:1 #ff7847)",
        "desc": "动作标准分析 · 训练效果评估",
        "features": ["动作识别", "姿势纠正", "训练计数", "强度评估"],
        "camera_title": "健身训练 · 实时监测",
        "camera_desc": "实时检测训练动作标准度、姿势风险与运动强度",
        "video_title": "健身训练 · 视频分析",
        "video_desc": "分析训练视频，评估动作质量、姿势风险与训练效果",
        "state_label": "训练状态",
        "count_label": "训练人数",
        "part_label": "训练活跃度",
        "stats_title": "训练统计面板",
        "stats_desc": "汇总历史训练数据 · 展示动作质量与训练趋势",
        "action_title": "训练动作识别统计",
        "behavior_title": "训练姿态识别统计",
    },
    "rehab": {
        "name": "康复训练",
        "icon": "🏥",
        "color": "#00d8a0",
        "color2": "#00b894",
        "gradient": "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00d8a0, stop:1 #00b894)",
        "desc": "活动范围监测 · 恢复进度追踪",
        "features": ["关节活动度", "动作范围", "恢复评估", "温和提示"],
        "camera_title": "康复训练 · 实时监测",
        "camera_desc": "实时监测关节活动范围、动作幅度与恢复进度",
        "video_title": "康复训练 · 视频分析",
        "video_desc": "分析康复视频，评估关节活动度、动作范围与恢复效果",
        "state_label": "康复状态",
        "count_label": "训练人数",
        "part_label": "活动完成度",
        "stats_title": "康复统计面板",
        "stats_desc": "汇总历史康复数据 · 展示活动范围与恢复趋势",
        "action_title": "康复动作识别统计",
        "behavior_title": "康复姿态识别统计",
    },
    "general": {
        "name": "通用模式",
        "icon": "⚡",
        "color": "#a855f7",
        "color2": "#7c3aed",
        "gradient": "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #a855f7, stop:1 #7c3aed)",
        "desc": "全功能检测 · 多场景适配",
        "features": ["全部检测项", "自定义配置", "灵活应用", "数据导出"],
        "camera_title": "通用监测 · 实时检测",
        "camera_desc": "全功能实时检测人员姿态、风险动作与运动行为",
        "video_title": "通用监测 · 视频分析",
        "video_desc": "全功能视频分析，检测所有动作、行为与风险项",
        "state_label": "检测状态",
        "count_label": "检测人数",
        "part_label": "活动参与度",
        "stats_title": "综合统计面板",
        "stats_desc": "汇总全部监测数据 · 多维度数据分析",
        "action_title": "运动动作识别统计",
        "behavior_title": "行为姿态识别统计",
    },
}


# ================================================================
# 全局样式表
# ================================================================
STYLE_SHEET = """
QMainWindow, QWidget#CentralWidget {
    background-color: #f0f2f5;
}

/* ===== 侧边栏 ===== */
#Sidebar {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #1a1a2e, stop:0.4 #16213e, stop:1 #0f3460);
    border-right: 1px solid rgba(0,0,0,0.3);
}
#LogoLabel {
    color: #ffffff;
    font-size: 17px;
    font-weight: bold;
    padding: 22px 18px 4px 18px;
}
#LogoSub {
    color: #8b8ba0;
    font-size: 10px;
    padding: 0px 18px 18px 18px;
}
#NavList {
    background: transparent;
    border: none;
    outline: none;
}
#NavList::item {
    color: #c8c8d8;
    font-size: 13px;
    padding: 14px 18px;
    border-left: 3px solid transparent;
    border-radius: 0px;
}
#NavList::item:hover {
    background: rgba(255,255,255,0.06);
    color: #ffffff;
}
#NavList::item:selected {
    background: rgba(79, 172, 254, 0.12);
    color: #4facfe;
    border-left: 3px solid #4facfe;
    font-weight: bold;
}
#VersionLabel {
    color: #6c7293;
    font-size: 10px;
    padding: 12px 18px;
}

/* ===== 场景选择栏 ===== */
#SceneBar {
    background: #ffffff;
    border-bottom: 1px solid #e8e8ef;
}
#SceneBtn {
    background: transparent;
    border: none;
    border-radius: 10px;
    padding: 10px 20px;
    font-size: 13px;
    font-weight: bold;
    color: #6c7293;
    text-align: left;
}
#SceneBtn:hover {
    background: #f0f4ff;
}
#SceneBtn:checked {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #4facfe, stop:1 #00f2fe);
    color: white;
}

/* ===== 内容区 ===== */
#Content {
    background-color: #f0f2f5;
}
#PageTitle {
    color: #1a1a2e;
    font-size: 22px;
    font-weight: bold;
}
#PageDesc {
    color: #6c7293;
    font-size: 12px;
}

/* ===== 卡片 ===== */
QFrame#Card {
    background-color: #ffffff;
    border-radius: 14px;
    border: 1px solid #eaecef;
}
QFrame#StatCard {
    background-color: #ffffff;
    border-radius: 12px;
    border: 1px solid #eaecef;
}
QFrame#GradientCard {
    border-radius: 14px;
    border: none;
}
QLabel#StatValue {
    font-size: 30px;
    font-weight: bold;
    color: #1a1a2e;
}
QLabel#StatLabel {
    font-size: 11px;
    color: #8b8ba0;
}
QLabel#StatIcon {
    font-size: 28px;
}

/* ===== 按钮 ===== */
QPushButton#Primary {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #4facfe, stop:1 #00f2fe);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 11px 28px;
    font-size: 14px;
    font-weight: bold;
}
QPushButton#Primary:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #3d8ce0, stop:1 #00d8e0);
}
QPushButton#Primary:disabled {
    background: #d0d0d8;
    color: #999;
}
QPushButton#Danger {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #ff416c, stop:1 #ff4b2b);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 11px 28px;
    font-size: 14px;
    font-weight: bold;
}
QPushButton#Danger:hover {
    background: #e83055;
}
QPushButton#Danger:disabled {
    background: #d0d0d8;
    color: #999;
}
QPushButton#Secondary {
    background-color: #ffffff;
    color: #4facfe;
    border: 2px solid #4facfe;
    border-radius: 10px;
    padding: 9px 22px;
    font-size: 13px;
    font-weight: bold;
}
QPushButton#Secondary:hover {
    background-color: #f0f8ff;
}
QPushButton#GhostBtn {
    background: transparent;
    color: #6c7293;
    border: 1px solid #e8e8ef;
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 12px;
}
QPushButton#GhostBtn:hover {
    background: #f8f9fc;
    color: #1a1a2e;
    border-color: #c8c8d8;
}

/* ===== 进度条 ===== */
QProgressBar {
    border: none;
    border-radius: 8px;
    background-color: #e8e8ef;
    text-align: center;
    font-size: 12px;
    color: #1a1a2e;
    height: 24px;
}
QProgressBar::chunk {
    border-radius: 8px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #4facfe, stop:1 #00f2fe);
}

/* ===== 表格 ===== */
QTableWidget {
    background-color: #ffffff;
    border: 1px solid #eaecef;
    border-radius: 10px;
    gridline-color: #f5f5f8;
    font-size: 13px;
}
QTableWidget::item {
    padding: 10px 8px;
    color: #1a1a2e;
}
QHeaderView::section {
    background-color: #f8f9fc;
    color: #6c7293;
    border: none;
    border-bottom: 1px solid #eaecef;
    padding: 12px 10px;
    font-weight: bold;
    font-size: 12px;
}
QTableWidget::item:selected {
    background-color: #e8f4ff;
    color: #1a1a2e;
}

/* ===== 视频显示区 ===== */
QLabel#VideoDisplay {
    background-color: #1a1a2e;
    border-radius: 12px;
    color: #6c7293;
    qproperty-alignment: AlignCenter;
    font-size: 14px;
}

/* ===== 拖拽区 ===== */
QFrame#DropZone {
    background-color: #f8faff;
    border: 3px dashed #c5d3e8;
    border-radius: 16px;
}
QFrame#DropZone:hover {
    border-color: #4facfe;
    background-color: #f0f6ff;
}
QFrame#DropZoneActive {
    background-color: #e8f4ff;
    border: 3px solid #4facfe;
    border-radius: 16px;
}
QLabel#DropIcon {
    font-size: 48px;
}
QLabel#DropText {
    font-size: 16px;
    font-weight: bold;
    color: #4facfe;
}
QLabel#DropHint {
    font-size: 12px;
    color: #8b8ba0;
}

/* ===== 列表 ===== */
QListWidget#RecentList {
    background-color: #ffffff;
    border: 1px solid #eaecef;
    border-radius: 10px;
    font-size: 13px;
    outline: none;
}
QListWidget#RecentList::item {
    padding: 12px 16px;
    border-bottom: 1px solid #f5f5f8;
    color: #1a1a2e;
}
QListWidget#RecentList::item:hover {
    background-color: #f8f9fc;
}
QListWidget#RecentList::item:selected {
    background-color: #e8f4ff;
    color: #1a1a2e;
}

/* ===== 滚动区 ===== */
QScrollArea {
    border: none;
    background: transparent;
}

/* ===== 组合框 ===== */
QComboBox {
    border: 2px solid #e8e8ef;
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 13px;
    background: white;
}
QComboBox:hover {
    border-color: #4facfe;
}
"""


# ================================================================
# 最近文件管理
# ================================================================
class RecentFilesManager:
    """管理最近打开的视频文件列表"""

    def __init__(self, max_count=8):
        self.max_count = max_count
        self.config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "recent_files.json"
        )
        self.files = self._load()

    def _load(self):
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _save(self):
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.files, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def add(self, path):
        if not path or not os.path.exists(path):
            return
        path = os.path.normpath(path)
        if path in self.files:
            self.files.remove(path)
        self.files.insert(0, path)
        self.files = self.files[:self.max_count]
        self._save()

    def remove(self, path):
        path = os.path.normpath(path)
        if path in self.files:
            self.files.remove(path)
            self._save()

    def clear(self):
        self.files = []
        self._save()

    def get_all(self):
        return [f for f in self.files if os.path.exists(f)]


# ================================================================
# 数据库管理
# ================================================================
class DatabaseManager:
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "monitoring_history.db"
            )
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                mode TEXT NOT NULL,
                scene TEXT DEFAULT 'campus',
                source TEXT,
                duration_sec REAL,
                total_frames INTEGER,
                avg_fps REAL,
                peak_persons INTEGER,
                avg_persons REAL,
                participation REAL,
                risk_alerts INTEGER,
                blaze_confirm INTEGER,
                action_dist TEXT,
                behavior_dist TEXT,
                report_path TEXT
            )
        """)
        # 自动迁移：检查是否有 scene 列，没有则重建表
        c.execute("PRAGMA table_info(records)")
        cols = [row[1] for row in c.fetchall()]
        if "scene" not in cols and len(cols) > 0:
            # 旧表缺少 scene 列，需要迁移
            c.execute("ALTER TABLE records RENAME TO records_old")
            c.execute("""
                CREATE TABLE records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    scene TEXT DEFAULT 'campus',
                    source TEXT,
                    duration_sec REAL,
                    total_frames INTEGER,
                    avg_fps REAL,
                    peak_persons INTEGER,
                    avg_persons REAL,
                    participation REAL,
                    risk_alerts INTEGER,
                    blaze_confirm INTEGER,
                    action_dist TEXT,
                    behavior_dist TEXT,
                    report_path TEXT
                )
            """)
            # 迁移旧数据（按旧列顺序映射）
            c.execute("""
                INSERT INTO records (timestamp, mode, scene, source, duration_sec,
                    total_frames, avg_fps, peak_persons, avg_persons,
                    participation, risk_alerts, blaze_confirm,
                    action_dist, behavior_dist, report_path)
                SELECT timestamp, mode, 'campus', source, duration_sec,
                    total_frames, avg_fps, peak_persons, avg_persons,
                    participation, risk_alerts, blaze_confirm,
                    action_dist, behavior_dist, report_path
                FROM records_old
            """)
            c.execute("DROP TABLE records_old")
        conn.commit()
        conn.close()

    def save_record(self, data):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            INSERT INTO records (timestamp, mode, scene, source, duration_sec,
                total_frames, avg_fps, peak_persons, avg_persons,
                participation, risk_alerts, blaze_confirm,
                action_dist, behavior_dist, report_path)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            data.get("mode", ""),
            data.get("scene", "campus"),
            data.get("source", ""),
            data.get("duration_sec", 0),
            data.get("total_frames", 0),
            data.get("avg_fps", 0),
            data.get("peak_persons", 0),
            data.get("avg_persons", 0),
            data.get("participation", 0),
            data.get("risk_alerts", 0),
            data.get("blaze_confirm", 0),
            json.dumps(data.get("action_dist", {}), ensure_ascii=False),
            json.dumps(data.get("behavior_dist", {}), ensure_ascii=False),
            data.get("report_path", ""),
        ))
        conn.commit()
        conn.close()

    def get_all_records(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT * FROM records ORDER BY timestamp DESC")
        rows = c.fetchall()
        conn.close()
        return rows

    def get_record(self, rid):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT * FROM records WHERE id=?", (rid,))
        row = c.fetchone()
        conn.close()
        return row

    def delete_record(self, rid):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("DELETE FROM records WHERE id=?", (rid,))
        conn.commit()
        conn.close()

    def clear_all(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("DELETE FROM records")
        conn.commit()
        conn.close()


# ================================================================
# 工作线程
# ================================================================
class VideoProcessThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    stats_ready = pyqtSignal(dict)
    progress_update = pyqtSignal(int, int)
    finished_report = pyqtSignal(dict)

    def __init__(self, video_path, output_path=None, use_blazepose=True):
        super().__init__()
        self.video_path = video_path
        self.output_path = output_path
        self.use_blazepose = use_blazepose
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        ok, err = load_engine()
        if not ok:
            self.stats_ready.emit({"error": f"检测引擎加载失败:\n{err[:300]}"})
            return
        self.system = SportsClassroomSystem(use_blazepose=self.use_blazepose)
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.stats_ready.emit({"error": "无法打开视频"})
            return

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 25
        writer = None
        if self.output_path:
            w_src = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            h_src = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(self.output_path, fourcc, fps_src, (w_src, h_src))

        total_frames = 0
        sum_total = sum_moving = sum_standing = sum_still = 0
        sum_alerts = sum_confirmed = 0
        max_total = 0
        action_dist = Counter()
        behavior_dist = Counter()
        prev_t = time.time()
        fps_smooth = 0.0
        start_time = time.time()

        while self._running:
            ok, frame = cap.read()
            if not ok:
                break

            frame, stats = self.system.process_frame(frame)
            now = time.time()
            inst_fps = 1.0 / max(now - prev_t, 1e-6)
            prev_t = now
            fps_smooth = 0.9 * fps_smooth + 0.1 * inst_fps if fps_smooth else inst_fps

            draw_panel(frame, stats, fps_smooth, self.system.use_blazepose)

            total_frames += 1
            sum_total += stats["total"]
            sum_moving += stats["moving"]
            sum_standing += stats["standing"]
            sum_still += stats["still"]
            sum_alerts += stats["alerts"]
            sum_confirmed += stats["confirmed"]
            if stats["total"] > max_total:
                max_total = stats["total"]

            for a in stats.get("cur_actions", []):
                action_dist[a] += 1
            for b in stats.get("cur_behaviors", []):
                behavior_dist[b] += 1

            if writer:
                writer.write(frame)

            if total_frames % 3 == 0 or total_frames == 1:
                self.frame_ready.emit(frame)
            self.progress_update.emit(total_frames, total)
            self.stats_ready.emit(stats)

        duration = time.time() - start_time
        cap.release()
        if writer:
            writer.release()
        if self.system:
            self.system.close()

        report = {
            "total_frames": total_frames,
            "duration_sec": round(duration, 1),
            "avg_fps": round(fps_smooth, 1),
            "peak_persons": max_total,
            "avg_persons": round(sum_total / max(total_frames, 1), 1),
            "participation": round((sum_moving + sum_standing) / max(sum_total, 1), 3),
            "risk_alerts": sum_alerts,
            "blaze_confirm": sum_confirmed,
            "action_dist": dict(action_dist),
            "behavior_dist": dict(behavior_dist),
            "output_path": self.output_path or "",
        }
        self.finished_report.emit(report)


class CameraThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    stats_ready = pyqtSignal(dict)

    def __init__(self, camera_index=0, use_blazepose=True):
        super().__init__()
        self.camera_index = camera_index
        self.use_blazepose = use_blazepose
        self._running = True
        self.system = None

    def stop(self):
        self._running = False

    def run(self):
        ok, err = load_engine()
        if not ok:
            self.stats_ready.emit({"error": f"检测引擎加载失败:\n{err[:300]}"})
            return
        self.system = SportsClassroomSystem(use_blazepose=self.use_blazepose)
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self.stats_ready.emit({"error": "无法打开摄像头"})
            return

        prev_t = time.time()
        fps_smooth = 0.0

        while self._running:
            ok, frame = cap.read()
            if not ok:
                continue

            frame, stats = self.system.process_frame(frame)
            now = time.time()
            inst_fps = 1.0 / max(now - prev_t, 1e-6)
            prev_t = now
            fps_smooth = 0.9 * fps_smooth + 0.1 * inst_fps if fps_smooth else inst_fps

            draw_panel(frame, stats, fps_smooth, self.system.use_blazepose)

            self.frame_ready.emit(frame)
            self.stats_ready.emit(stats)

        cap.release()
        if self.system:
            self.system.close()


# ================================================================
# 自定义组件：拖拽视频区
# ================================================================
class DragDropVideoArea(QFrame):
    """支持拖拽和点击选择的视频导入区域"""
    file_dropped = pyqtSignal(str)

    def __init__(self, recent_manager=None):
        super().__init__()
        self.recent = recent_manager
        self.setAcceptDrops(True)
        self.setObjectName("DropZone")
        self.setMinimumHeight(200)
        self.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(8)

        self.icon_label = QLabel("📁")
        self.icon_label.setObjectName("DropIcon")
        self.icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.icon_label)

        self.text_label = QLabel("拖拽视频文件到此处  或  点击选择")
        self.text_label.setObjectName("DropText")
        self.text_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.text_label)

        self.hint_label = QLabel("支持 MP4 / AVI / MOV / MKV  ·  也可从下方最近文件中选择")
        self.hint_label.setObjectName("DropHint")
        self.hint_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.hint_label)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            for url in urls:
                path = url.toLocalFile()
                if path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv')):
                    event.acceptProposedAction()
                    self.setObjectName("DropZoneActive")
                    self.style().unpolish(self)
                    self.style().polish(self)
                    self.icon_label.setText("📥")
                    return
        event.ignore()

    def dragLeaveEvent(self, event):
        self.setObjectName("DropZone")
        self.style().unpolish(self)
        self.style().polish(self)
        self.icon_label.setText("📁")

    def dropEvent(self, event):
        self.setObjectName("DropZone")
        self.style().unpolish(self)
        self.style().polish(self)
        self.icon_label.setText("📁")
        urls = event.mimeData().urls()
        for url in urls:
            path = url.toLocalFile()
            if path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv')):
                self.file_dropped.emit(path)
                return

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            path, _ = QFileDialog.getOpenFileName(
                self, "选择视频文件", "",
                "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv);;所有文件 (*)")
            if path:
                self.file_dropped.emit(path)


# ================================================================
# 自定义组件：带阴影效果的卡片
# ================================================================
def add_shadow(widget, blur=20, offset=2, color=QColor(0, 0, 0, 25)):
    effect = QGraphicsDropShadowEffect(widget)
    effect.setBlurRadius(blur)
    effect.setOffset(offset, offset)
    effect.setColor(color)
    widget.setGraphicsEffect(effect)


# ================================================================
# 页面：数据仪表盘
# ================================================================
class DashboardPage(QWidget):
    def __init__(self, db_manager, recent_manager, scene_changed_callback=None):
        super().__init__()
        self.db = db_manager
        self.recent = recent_manager
        self.scene_callback = scene_changed_callback
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        # 标题
        title_col = QVBoxLayout()
        title_col.setSpacing(4)
        title = QLabel("数据仪表盘")
        title.setObjectName("PageTitle")
        title_col.addWidget(title)
        desc = QLabel("总览监测数据 · 快速选择场景 · 查看最近记录")
        desc.setObjectName("PageDesc")
        title_col.addWidget(desc)
        layout.addLayout(title_col)

        # 场景快速选择卡片
        scene_row = QHBoxLayout()
        scene_row.setSpacing(14)
        for key, cfg in SCENE_CONFIG.items():
            card = self._make_scene_card(key, cfg)
            scene_row.addWidget(card)
        layout.addLayout(scene_row)

        # 汇总统计卡片行
        stats_row = QHBoxLayout()
        stats_row.setSpacing(14)
        self.lbl_sessions = self._make_stat_card(stats_row, "📋", "监测总次数", "0", "#4facfe")
        self.lbl_frames = self._make_stat_card(stats_row, "🎬", "总分析帧数", "0", "#00d8a0")
        self.lbl_alerts = self._make_stat_card(stats_row, "⚠️", "累计风险告警", "0", "#ff4b4b")
        self.lbl_part = self._make_stat_card(stats_row, "📊", "平均参与度", "0%", "#ff9f43")
        layout.addLayout(stats_row)

        # 最近文件 + 快速操作
        body = QHBoxLayout()
        body.setSpacing(16)

        # 最近视频
        recent_group = QFrame()
        recent_group.setObjectName("Card")
        add_shadow(recent_group)
        rl = QVBoxLayout(recent_group)
        rl.setContentsMargins(20, 20, 20, 20)
        rl.setSpacing(12)
        rtitle = QLabel("🕐 最近视频文件")
        rtitle.setStyleSheet("font-size:15px;font-weight:bold;color:#1a1a2e;")
        rl.addWidget(rtitle)

        self.recent_list = QListWidget()
        self.recent_list.setObjectName("RecentList")
        self.recent_list.setMaximumHeight(220)
        rl.addWidget(self.recent_list)

        btn_row = QHBoxLayout()
        self.btn_clear_recent = QPushButton("清空列表")
        self.btn_clear_recent.setObjectName("GhostBtn")
        self.btn_clear_recent.clicked.connect(self.clear_recent)
        btn_row.addWidget(self.btn_clear_recent)
        btn_row.addStretch()
        rl.addLayout(btn_row)
        body.addWidget(recent_group, 1)

        # 快速操作
        action_group = QFrame()
        action_group.setObjectName("Card")
        add_shadow(action_group)
        al = QVBoxLayout(action_group)
        al.setContentsMargins(20, 20, 20, 20)
        al.setSpacing(14)
        atitle = QLabel("🚀 快速操作")
        atitle.setStyleSheet("font-size:15px;font-weight:bold;color:#1a1a2e;")
        al.addWidget(atitle)

        for icon, text, desc, color in [
            ("📷", "实时摄像头监测", "打开摄像头实时检测", "#4facfe"),
            ("🎬", "视频文件分析", "拖拽或选择视频分析", "#ff9f43"),
            ("📊", "查看统计面板", "汇总历史监测数据", "#00d8a0"),
            ("📋", "历史记录管理", "查看删除导出记录", "#a855f7"),
        ]:
            btn = QPushButton(f"  {icon}  {text}")
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: {color}15;
                    color: {color};
                    border: 1px solid {color}40;
                    border-radius: 10px;
                    padding: 12px 16px;
                    font-size: 13px;
                    font-weight: bold;
                    text-align: left;
                }}
                QPushButton:hover {{
                    background: {color}25;
                    border-color: {color};
                }}
            """)
            al.addWidget(btn)
        al.addStretch()
        body.addWidget(action_group, 1)

        layout.addLayout(body, 1)
        self.load_data()

    def _make_scene_card(self, key, cfg):
        card = QFrame()
        card.setObjectName("GradientCard")
        card.setMinimumHeight(120)
        card.setMaximumHeight(130)
        card.setCursor(Qt.PointingHandCursor)
        card.setStyleSheet(f"""
            QFrame#GradientCard {{
                background: {cfg['gradient']};
                border-radius: 14px;
            }}
            QFrame#GradientCard:hover {{
                border: 2px solid white;
            }}
        """)

        cl = QVBoxLayout(card)
        cl.setContentsMargins(18, 16, 18, 16)
        cl.setSpacing(6)

        icon = QLabel(f"{cfg['icon']}  {cfg['name']}")
        icon.setStyleSheet("color:white;font-size:16px;font-weight:bold;")
        cl.addWidget(icon)

        desc = QLabel(cfg["desc"])
        desc.setStyleSheet("color:rgba(255,255,255,0.85);font-size:11px;")
        desc.setWordWrap(True)
        cl.addWidget(desc)

        feats = QLabel("  ·  ".join(cfg["features"][:2]))
        feats.setStyleSheet("color:rgba(255,255,255,0.65);font-size:10px;")
        cl.addWidget(feats)

        def on_click(event, k=key):
            if self.scene_callback:
                self.scene_callback(k)

        card.mousePressEvent = on_click
        add_shadow(card, blur=15, offset=3, color=QColor(0, 0, 0, 30))
        return card

    def _make_stat_card(self, parent_layout, icon, label, value, color):
        card = QFrame()
        card.setObjectName("StatCard")
        card.setMinimumHeight(100)
        card.setMaximumHeight(110)
        add_shadow(card)
        l = QHBoxLayout(card)
        l.setContentsMargins(20, 16, 20, 16)
        l.setSpacing(14)

        icon_lbl = QLabel(icon)
        icon_lbl.setStyleSheet(f"""
            font-size:32px;
            background: {color}15;
            border-radius: 10px;
            padding: 8px;
        """)
        l.addWidget(icon_lbl)

        info = QVBoxLayout()
        info.setSpacing(2)
        lbl_val = QLabel(value)
        lbl_val.setObjectName("StatValue")
        lbl_val.setStyleSheet(f"color:{color};")
        lbl_label = QLabel(label)
        lbl_label.setObjectName("StatLabel")
        info.addWidget(lbl_val)
        info.addWidget(lbl_label)
        l.addLayout(info)
        l.addStretch()
        parent_layout.addWidget(card)
        return lbl_val

    def load_data(self):
        records = self.db.get_all_records()
        total_sessions = len(records)
        # 新结构: r[6]=total_frames, r[11]=risk_alerts, r[10]=participation
        total_frames = sum(r[6] for r in records if r[6])
        total_alerts = sum(r[11] for r in records if r[11])
        avg_part = sum(r[10] for r in records if r[10]) / max(total_sessions, 1)

        self.lbl_sessions.setText(str(total_sessions))
        self.lbl_frames.setText(str(total_frames))
        self.lbl_alerts.setText(str(total_alerts))
        self.lbl_part.setText(f"{avg_part:.0%}")

        # 最近文件
        self.recent_list.clear()
        for f in self.recent.get_all():
            name = os.path.basename(f)
            dirname = os.path.dirname(f)
            item = QListWidgetItem(f"🎬  {name}\n    📂 {dirname}")
            item.setSizeHint(QSize(0, 50))
            self.recent_list.addItem(item)

    def clear_recent(self):
        self.recent.clear()
        self.load_data()


# ================================================================
# 页面：实时摄像头监测
# ================================================================
class CameraPage(QWidget):
    def __init__(self, db_manager, current_scene="campus"):
        super().__init__()
        self.db = db_manager
        self.current_scene = current_scene
        self.camera_thread = None
        self.start_time = None
        self.frame_count = 0
        self.sum_alerts = 0
        self.action_dist = Counter()
        self.behavior_dist = Counter()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        title_col = QVBoxLayout()
        title_col.setSpacing(4)
        self.title_label = QLabel("实时摄像头监测")
        self.title_label.setObjectName("PageTitle")
        title_col.addWidget(self.title_label)
        self.desc_label = QLabel("打开摄像头实时检测人员姿态、风险动作与课堂行为")
        self.desc_label.setObjectName("PageDesc")
        title_col.addWidget(self.desc_label)
        layout.addLayout(title_col)

        # 统计卡片
        stats_row = QHBoxLayout()
        stats_row.setSpacing(12)
        self.lbl_count, self.lbl_count_title = self._make_stat(stats_row, "👥", "检测人数", "0", "#4facfe")
        self.lbl_part, self.lbl_part_title = self._make_stat(stats_row, "📈", "参与度", "0%", "#00d8a0")
        self.lbl_alerts, _ = self._make_stat(stats_row, "⚠️", "风险告警", "0", "#ff4b4b")
        self.lbl_fps, _ = self._make_stat(stats_row, "⚡", "处理帧率", "0.0", "#ff9f43")
        layout.addLayout(stats_row)

        # 主体
        body = QHBoxLayout()
        body.setSpacing(16)

        self.video_label = QLabel("点击「开始监测」启动摄像头\n\n系统将实时检测画面中的人员姿态")
        self.video_label.setObjectName("VideoDisplay")
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        body.addWidget(self.video_label, 3)

        # 侧边面板
        side = QVBoxLayout()
        side.setSpacing(12)

        state_group = QFrame()
        state_group.setObjectName("Card")
        add_shadow(state_group)
        sl = QVBoxLayout(state_group)
        sl.setContentsMargins(18, 18, 18, 18)
        sl.setSpacing(8)
        self.st_title = QLabel("📊 课堂状态")
        self.st_title.setStyleSheet("font-size:14px;font-weight:bold;color:#1a1a2e;")
        sl.addWidget(self.st_title)
        self.lbl_states = QLabel("运动: 0  站立: 0  静止: 0")
        self.lbl_states.setStyleSheet("font-size:13px;color:#6c7293;padding:4px 0;")
        sl.addWidget(self.lbl_states)
        self.lbl_action = QLabel("动作: 无")
        self.lbl_action.setStyleSheet("font-size:13px;color:#00d8a0;padding:4px 0;")
        sl.addWidget(self.lbl_action)
        self.lbl_behavior = QLabel("行为: 无")
        self.lbl_behavior.setStyleSheet("font-size:13px;color:#ff9f43;padding:4px 0;")
        sl.addWidget(self.lbl_behavior)
        self.lbl_time = QLabel("时长: 00:00")
        self.lbl_time.setStyleSheet("font-size:13px;color:#6c7293;padding:4px 0;")
        sl.addWidget(self.lbl_time)
        side.addWidget(state_group)

        # 控制按钮
        btn_layout = QVBoxLayout()
        btn_layout.setSpacing(10)
        self.btn_start = QPushButton("▶  开始监测")
        self.btn_start.setObjectName("Primary")
        self.btn_start.clicked.connect(self.start_camera)
        btn_layout.addWidget(self.btn_start)

        self.btn_stop = QPushButton("■  停止监测")
        self.btn_stop.setObjectName("Danger")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_camera)
        btn_layout.addWidget(self.btn_stop)

        self.btn_blazepose = QPushButton("BlazePose 精细分析: 开启")
        self.btn_blazepose.setObjectName("Secondary")
        self.btn_blazepose.setCheckable(True)
        self.btn_blazepose.setChecked(True)
        btn_layout.addWidget(self.btn_blazepose)

        side.addLayout(btn_layout)
        side.addStretch()
        body.addLayout(side, 1)
        layout.addLayout(body, 1)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_time)

    def _make_stat(self, parent_layout, icon, label, value, color):
        card = QFrame()
        card.setObjectName("StatCard")
        card.setMinimumHeight(100)
        card.setMaximumHeight(110)
        add_shadow(card)
        l = QHBoxLayout(card)
        l.setContentsMargins(20, 16, 20, 16)
        l.setSpacing(12)
        icon_lbl = QLabel(icon)
        icon_lbl.setStyleSheet(f"font-size:28px;background:{color}15;border-radius:8px;padding:6px;")
        l.addWidget(icon_lbl)
        info = QVBoxLayout()
        info.setSpacing(2)
        lbl_val = QLabel(value)
        lbl_val.setObjectName("StatValue")
        lbl_val.setStyleSheet(f"color:{color};")
        lbl_label = QLabel(label)
        lbl_label.setObjectName("StatLabel")
        info.addWidget(lbl_val)
        info.addWidget(lbl_label)
        l.addLayout(info)
        l.addStretch()
        parent_layout.addWidget(card)
        return lbl_val, lbl_label

    def update_scene(self, scene_key):
        """根据场景模式更新页面文案"""
        self.current_scene = scene_key
        cfg = SCENE_CONFIG.get(scene_key, SCENE_CONFIG["campus"])
        self.title_label.setText(cfg["camera_title"])
        self.desc_label.setText(cfg["camera_desc"])
        self.st_title.setText(f"📊 {cfg['state_label']}")
        self.lbl_count_title.setText(cfg["count_label"])
        self.lbl_part_title.setText(cfg["part_label"])
        self.video_label.setText(f"点击「开始监测」启动摄像头\n\n{cfg['camera_desc']}")

    def start_camera(self):
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.frame_count = 0
        self.sum_alerts = 0
        self.sum_total = 0
        self.sum_moving = 0
        self.sum_standing = 0
        self.sum_still = 0
        self.peak_persons = 0
        self.action_dist.clear()
        self.behavior_dist.clear()
        self.start_time = time.time()

        use_bp = self.btn_blazepose.isChecked()
        self.camera_thread = CameraThread(0, use_bp)
        self.camera_thread.frame_ready.connect(self.on_frame)
        self.camera_thread.stats_ready.connect(self.on_stats)
        self.camera_thread.start()
        self.timer.start(1000)

    def stop_camera(self):
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread.wait()
            self.camera_thread = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.timer.stop()

        if self.frame_count > 0:
            duration = time.time() - self.start_time if self.start_time else 0
            # 计算参与度：(运动+站立) / 总人数，按帧累计平均
            participation = round(
                (self.sum_moving + self.sum_standing) / max(self.sum_total, 1), 3)
            avg_persons = round(self.sum_total / max(self.frame_count, 1), 1)
            avg_fps = round(self.frame_count / max(duration, 1), 1)
            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "mode": "实时摄像头",
                "scene": self.current_scene,
                "source": "摄像头0",
                "duration_sec": round(duration, 1),
                "total_frames": self.frame_count,
                "avg_fps": avg_fps,
                "peak_persons": self.peak_persons,
                "avg_persons": avg_persons,
                "participation": participation,
                "risk_alerts": self.sum_alerts,
                "blaze_confirm": 0,
                "action_dist": dict(self.action_dist),
                "behavior_dist": dict(self.behavior_dist),
                "report_path": "",
            }
            self.db.save_record(record)
            QMessageBox.information(self, "监测结束",
                f"本次监测已保存到历史记录\n\n"
                f"监测时长: {duration:.0f}秒\n"
                f"处理帧数: {self.frame_count}\n"
                f"峰值人数: {self.peak_persons}人\n"
                f"平均人数: {avg_persons}人/帧\n"
                f"参与度: {participation:.0%}\n"
                f"风险告警: {self.sum_alerts}次")

    def on_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pix)
        self.frame_count += 1

    def on_stats(self, stats):
        if "error" in stats:
            self.video_label.setText(stats["error"])
            self.stop_camera()
            return
        total = stats.get("total", 0)
        moving = stats.get("moving", 0)
        standing = stats.get("standing", 0)
        still = stats.get("still", 0)
        # 累计统计用于计算参与度
        self.sum_total += total
        self.sum_moving += moving
        self.sum_standing += standing
        self.sum_still += still
        if total > self.peak_persons:
            self.peak_persons = total

        self.lbl_count.setText(str(total))
        part = (moving + standing) / max(total, 1) if total > 0 else 0
        self.lbl_part.setText(f"{part:.0%}")
        self.sum_alerts += stats.get("alerts", 0)
        self.lbl_alerts.setText(str(self.sum_alerts))
        self.lbl_states.setText(
            f"运动: {moving}  站立: {standing}  静止: {still}")
        actions = stats.get("cur_actions", [])
        self.lbl_action.setText(f"动作: {', '.join(actions) if actions else '无'}")
        for a in actions:
            self.action_dist[a] += 1
        behaviors = stats.get("cur_behaviors", [])
        self.lbl_behavior.setText(f"行为: {', '.join(behaviors) if behaviors else '无'}")
        for b in behaviors:
            self.behavior_dist[b] += 1

    def update_time(self):
        if self.start_time:
            elapsed = int(time.time() - self.start_time)
            m, s = divmod(elapsed, 60)
            self.lbl_time.setText(f"时长: {m:02d}:{s:02d}")
            self.lbl_fps.setText(f"{self.frame_count / max(elapsed,1):.1f}")


# ================================================================
# 页面：视频文件分析（支持拖拽 + 最近文件）
# ================================================================
class VideoAnalyzePage(QWidget):
    def __init__(self, db_manager, recent_manager, current_scene="campus"):
        super().__init__()
        self.db = db_manager
        self.recent = recent_manager
        self.current_scene = current_scene
        self.process_thread = None
        self.video_path = ""
        self.sum_alerts = 0
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        title_col = QVBoxLayout()
        title_col.setSpacing(4)
        self.title_label = QLabel("视频文件分析")
        self.title_label.setObjectName("PageTitle")
        title_col.addWidget(self.title_label)
        self.desc_label = QLabel("拖拽视频文件到下方区域  ·  或点击选择  ·  或从最近文件中选取")
        self.desc_label.setObjectName("PageDesc")
        title_col.addWidget(self.desc_label)
        layout.addLayout(title_col)

        # 拖拽导入区 + 最近文件 并排
        top_row = QHBoxLayout()
        top_row.setSpacing(16)

        # 拖拽区
        self.drop_area = DragDropVideoArea(self.recent)
        self.drop_area.file_dropped.connect(self.on_file_selected)
        top_row.addWidget(self.drop_area, 3)

        # 最近文件列表
        recent_group = QFrame()
        recent_group.setObjectName("Card")
        add_shadow(recent_group)
        rl = QVBoxLayout(recent_group)
        rl.setContentsMargins(16, 16, 16, 16)
        rl.setSpacing(8)
        rtitle = QLabel("🕐 最近文件")
        rtitle.setStyleSheet("font-size:13px;font-weight:bold;color:#1a1a2e;")
        rl.addWidget(rtitle)

        self.recent_list = QListWidget()
        self.recent_list.setObjectName("RecentList")
        self.recent_list.setMaximumWidth(280)
        self.recent_list.itemDoubleClicked.connect(self.on_recent_clicked)
        rl.addWidget(self.recent_list)
        top_row.addWidget(recent_group, 1)

        layout.addLayout(top_row)

        # 当前文件 + 控制按钮
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(12)
        self.lbl_file = QLabel("未选择文件")
        self.lbl_file.setStyleSheet(
            "background:white;border:2px solid #eaecef;border-radius:10px;"
            "padding:10px 16px;font-size:13px;color:#8b8ba0;")
        self.lbl_file.setMinimumWidth(300)
        ctrl_row.addWidget(self.lbl_file, 1)

        self.btn_start = QPushButton("▶  开始分析")
        self.btn_start.setObjectName("Primary")
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self.start_analysis)
        ctrl_row.addWidget(self.btn_start)

        self.btn_stop = QPushButton("■  停止")
        self.btn_stop.setObjectName("Danger")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_analysis)
        ctrl_row.addWidget(self.btn_stop)
        layout.addLayout(ctrl_row)

        # 进度条
        self.progress = QProgressBar()
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        # 主体：视频预览 + 统计
        body = QHBoxLayout()
        body.setSpacing(16)

        self.video_label = QLabel("选择视频文件后点击「开始分析」\n\n系统将逐帧检测并生成结果")
        self.video_label.setObjectName("VideoDisplay")
        self.video_label.setMinimumSize(480, 360)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        body.addWidget(self.video_label, 3)

        side = QVBoxLayout()
        side.setSpacing(10)

        stat_group = QFrame()
        stat_group.setObjectName("Card")
        add_shadow(stat_group)
        sl = QVBoxLayout(stat_group)
        sl.setContentsMargins(18, 18, 18, 18)
        sl.setSpacing(8)
        sl_title = QLabel("📊 分析进度")
        sl_title.setStyleSheet("font-size:14px;font-weight:bold;color:#1a1a2e;")
        sl.addWidget(sl_title)

        self.lbl_detail = QLabel("等待开始...")
        self.lbl_detail.setStyleSheet("font-size:13px;color:#6c7293;")
        sl.addWidget(self.lbl_detail)
        self.lbl_frame = QLabel("帧: 0 / 0")
        self.lbl_frame.setStyleSheet("font-size:13px;color:#6c7293;")
        sl.addWidget(self.lbl_frame)
        self.lbl_alerts = QLabel("累计告警: 0")
        self.lbl_alerts.setStyleSheet("font-size:13px;color:#ff4b4b;")
        sl.addWidget(self.lbl_alerts)
        side.addWidget(stat_group)
        side.addStretch()
        body.addLayout(side, 1)
        layout.addLayout(body, 1)

        self.refresh_recent()

    def update_scene(self, scene_key):
        """根据场景模式更新页面文案"""
        self.current_scene = scene_key
        cfg = SCENE_CONFIG.get(scene_key, SCENE_CONFIG["campus"])
        self.title_label.setText(cfg["video_title"])
        self.desc_label.setText(f"{cfg['video_desc']}  ·  拖拽或选择视频文件")
        self.video_label.setText(f"选择视频文件后点击「开始分析」\n\n{cfg['video_desc']}")

    def refresh_recent(self):
        self.recent_list.clear()
        for f in self.recent.get_all():
            name = os.path.basename(f)
            item = QListWidgetItem(f"🎬  {name}")
            item.setToolTip(f)
            self.recent_list.addItem(item)

    def on_recent_clicked(self, item):
        path = item.toolTip()
        if path and os.path.exists(path):
            self.on_file_selected(path)

    def on_file_selected(self, path):
        self.video_path = path
        name = os.path.basename(path)
        self.lbl_file.setText(f"✅  {name}")
        self.lbl_file.setStyleSheet(
            "background:white;border:2px solid #4facfe;border-radius:10px;"
            "padding:10px 16px;font-size:13px;color:#1a1a2e;")
        self.btn_start.setEnabled(True)
        self.recent.add(path)
        self.refresh_recent()

    def start_analysis(self):
        if not self.video_path or not os.path.exists(self.video_path):
            QMessageBox.warning(self, "错误", "视频文件不存在")
            return

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress.setValue(0)
        self.sum_alerts = 0

        out_dir = os.path.dirname(self.video_path)
        out_name = "result_" + os.path.basename(self.video_path)
        out_path = os.path.join(out_dir, out_name)

        self.process_thread = VideoProcessThread(self.video_path, out_path, True)
        self.process_thread.frame_ready.connect(self.on_frame)
        self.process_thread.stats_ready.connect(self.on_stats)
        self.process_thread.progress_update.connect(self.on_progress)
        self.process_thread.finished_report.connect(self.on_finished)
        self.process_thread.start()

    def stop_analysis(self):
        if self.process_thread:
            self.process_thread.stop()
            self.process_thread.wait()
            self.process_thread = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def on_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pix)

    def on_stats(self, stats):
        if "error" in stats:
            QMessageBox.warning(self, "错误", stats["error"])
            self.stop_analysis()
            return
        self.sum_alerts += stats.get("alerts", 0)
        self.lbl_alerts.setText(f"累计告警: {self.sum_alerts}")

    def on_progress(self, current, total):
        if total > 0:
            pct = int(current / total * 100)
            self.progress.setValue(pct)
            self.lbl_frame.setText(f"帧: {current} / {total}")
            self.lbl_detail.setText(f"分析中... {pct}%")

    def on_finished(self, report):
        self.progress.setValue(100)
        self.lbl_detail.setText("分析完成!")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "视频分析",
            "scene": self.current_scene,
            "source": os.path.basename(self.video_path),
            **report,
        }
        self.db.save_record(record)

        # 将结果视频加入最近文件列表
        output_path = report.get("output_path", "")
        if output_path and os.path.exists(output_path):
            self.recent.add(output_path)
            self.refresh_recent()

        msg = (
            f"视频分析完成!\n\n"
            f"总帧数: {report['total_frames']}\n"
            f"耗时: {report['duration_sec']}秒\n"
            f"平均帧率: {report['avg_fps']} FPS\n"
            f"峰值人数: {report['peak_persons']}人\n"
            f"平均人数: {report['avg_persons']}人/帧\n"
            f"参与度: {report['participation']:.1%}\n"
            f"风险告警: {report['risk_alerts']}次\n"
            f"BlazePose确认: {report['blaze_confirm']}次\n\n"
            f"结果视频已保存到:\n{report.get('output_path','')}\n\n"
            f"记录已存入历史记录。"
        )
        QMessageBox.information(self, "分析完成", msg)


# ================================================================
# 页面：课堂统计面板
# ================================================================
class StatsPanelPage(QWidget):
    def __init__(self, db_manager):
        super().__init__()
        self.db = db_manager
        self.current_scene = "campus"
        self.init_ui()

    def update_scene(self, scene_key):
        """根据场景模式更新页面文案"""
        self.current_scene = scene_key
        cfg = SCENE_CONFIG.get(scene_key, SCENE_CONFIG["campus"])
        self.title_label.setText(cfg["stats_title"])
        self.desc_label.setText(cfg["stats_desc"])
        self.al_title.setText(f"🏃 {cfg['action_title']}")
        self.bl_title.setText(f"👤 {cfg['behavior_title']}")
        self.load_data()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        title_col = QVBoxLayout()
        title_col.setSpacing(4)
        self.title_label = QLabel("课堂统计面板")
        self.title_label.setObjectName("PageTitle")
        title_col.addWidget(self.title_label)
        self.desc_label = QLabel("汇总历史监测数据 · 展示课堂质量与安全趋势")
        self.desc_label.setObjectName("PageDesc")
        title_col.addWidget(self.desc_label)
        layout.addLayout(title_col)

        # 汇总卡片
        summary_row = QHBoxLayout()
        summary_row.setSpacing(12)
        self.lbl_sessions = self._make_card(summary_row, "📋", "监测总次数", "0", "#4facfe")
        self.lbl_frames = self._make_card(summary_row, "🎬", "总分析帧数", "0", "#00d8a0")
        self.lbl_alerts = self._make_card(summary_row, "⚠️", "累计风险告警", "0", "#ff4b4b")
        self.lbl_part = self._make_card(summary_row, "📊", "平均参与度", "0%", "#ff9f43")
        layout.addLayout(summary_row)

        # 刷新
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.btn_refresh = QPushButton("🔄 刷新数据")
        self.btn_refresh.setObjectName("Secondary")
        self.btn_refresh.clicked.connect(self.load_data)
        btn_row.addWidget(self.btn_refresh)
        layout.addLayout(btn_row)

        # 统计表格
        body = QHBoxLayout()
        body.setSpacing(16)

        action_group = QFrame()
        action_group.setObjectName("Card")
        add_shadow(action_group)
        al = QVBoxLayout(action_group)
        al.setContentsMargins(18, 18, 18, 18)
        self.al_title = QLabel("🏃 运动动作识别统计")
        self.al_title.setStyleSheet("font-size:15px;font-weight:bold;color:#1a1a2e;")
        al.addWidget(self.al_title)
        self.action_table = QTableWidget(0, 2)
        self.action_table.setHorizontalHeaderLabels(["动作类型", "出现人次"])
        self.action_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        al.addWidget(self.action_table)
        body.addWidget(action_group, 1)

        behavior_group = QFrame()
        behavior_group.setObjectName("Card")
        add_shadow(behavior_group)
        bl = QVBoxLayout(behavior_group)
        bl.setContentsMargins(18, 18, 18, 18)
        self.bl_title = QLabel("👤 课堂行为识别统计")
        self.bl_title.setStyleSheet("font-size:15px;font-weight:bold;color:#1a1a2e;")
        bl.addWidget(self.bl_title)
        self.behavior_table = QTableWidget(0, 2)
        self.behavior_table.setHorizontalHeaderLabels(["行为类型", "出现人次"])
        self.behavior_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        bl.addWidget(self.behavior_table)
        body.addWidget(behavior_group, 1)
        layout.addLayout(body, 1)

        # 最近记录
        recent_group = QFrame()
        recent_group.setObjectName("Card")
        add_shadow(recent_group)
        rl = QVBoxLayout(recent_group)
        rl.setContentsMargins(18, 18, 18, 18)
        rl_title = QLabel("📋 最近监测记录")
        rl_title.setStyleSheet("font-size:15px;font-weight:bold;color:#1a1a2e;")
        rl.addWidget(rl_title)
        self.recent_table = QTableWidget(0, 6)
        self.recent_table.setHorizontalHeaderLabels(
            ["时间", "模式", "来源", "时长(秒)", "风险告警", "参与度"])
        self.recent_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        rl.addWidget(self.recent_table)
        layout.addWidget(recent_group)

    def _make_card(self, parent_layout, icon, label, value, color):
        card = QFrame()
        card.setObjectName("StatCard")
        card.setMinimumHeight(100)
        card.setMaximumHeight(110)
        add_shadow(card)
        l = QHBoxLayout(card)
        l.setContentsMargins(20, 16, 20, 16)
        l.setSpacing(12)
        icon_lbl = QLabel(icon)
        icon_lbl.setStyleSheet(f"font-size:28px;background:{color}15;border-radius:8px;padding:6px;")
        l.addWidget(icon_lbl)
        info = QVBoxLayout()
        info.setSpacing(2)
        lbl_val = QLabel(value)
        lbl_val.setObjectName("StatValue")
        lbl_val.setStyleSheet(f"color:{color};")
        lbl_label = QLabel(label)
        lbl_label.setObjectName("StatLabel")
        info.addWidget(lbl_val)
        info.addWidget(lbl_label)
        l.addLayout(info)
        l.addStretch()
        parent_layout.addWidget(card)
        return lbl_val

    def load_data(self):
        records = self.db.get_all_records()
        if not records:
            # 清空表格显示
            self.recent_table.setRowCount(0)
            self.action_table.setRowCount(0)
            self.behavior_table.setRowCount(0)
            return
        total_sessions = len(records)
        # 新结构: r[6]=total_frames, r[11]=risk_alerts, r[10]=participation
        total_frames = sum(r[6] for r in records if r[6])
        total_alerts = sum(r[11] for r in records if r[11])
        avg_part = sum(r[10] for r in records if r[10]) / max(total_sessions, 1)

        self.lbl_sessions.setText(str(total_sessions))
        self.lbl_frames.setText(str(total_frames))
        self.lbl_alerts.setText(str(total_alerts))
        self.lbl_part.setText(f"{avg_part:.0%}")

        action_sum = Counter()
        behavior_sum = Counter()
        for r in records:
            try:
                # 新结构: r[13]=action_dist, r[14]=behavior_dist
                action_sum.update(json.loads(r[13]) if r[13] else {})
                behavior_sum.update(json.loads(r[14]) if r[14] else {})
            except Exception:
                pass

        self._fill_table(self.action_table, action_sum)
        self._fill_table(self.behavior_table, behavior_sum)

        self.recent_table.setRowCount(min(len(records), 10))
        for i, r in enumerate(records[:10]):
            self.recent_table.setItem(i, 0, QTableWidgetItem(r[1] or ""))        # 时间
            self.recent_table.setItem(i, 1, QTableWidgetItem(r[2] or ""))        # 模式
            self.recent_table.setItem(i, 2, QTableWidgetItem(r[4] or ""))        # 来源
            self.recent_table.setItem(i, 3, QTableWidgetItem(str(r[5] or 0)))    # 时长
            self.recent_table.setItem(i, 4, QTableWidgetItem(str(r[11] or 0)))   # 风险告警
            self.recent_table.setItem(i, 5, QTableWidgetItem(f"{r[10] or 0:.0%}"))  # 参与度

    def _fill_table(self, table, data):
        items = sorted(data.items(), key=lambda x: x[1], reverse=True)
        table.setRowCount(len(items))
        for i, (k, v) in enumerate(items):
            table.setItem(i, 0, QTableWidgetItem(k))
            table.setItem(i, 1, QTableWidgetItem(str(v)))


# ================================================================
# 页面：历史记录管理
# ================================================================
class HistoryPage(QWidget):
    def __init__(self, db_manager):
        super().__init__()
        self.db = db_manager
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        title_col = QVBoxLayout()
        title_col.setSpacing(4)
        title = QLabel("历史记录管理")
        title.setObjectName("PageTitle")
        title_col.addWidget(title)
        desc = QLabel("查看 · 删除 · 导出历史监测记录")
        desc.setObjectName("PageDesc")
        title_col.addWidget(desc)
        layout.addLayout(title_col)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.btn_refresh = QPushButton("🔄 刷新")
        self.btn_refresh.setObjectName("Secondary")
        self.btn_refresh.clicked.connect(self.load_data)
        btn_row.addWidget(self.btn_refresh)

        self.btn_delete = QPushButton("🗑 删除选中")
        self.btn_delete.setObjectName("Danger")
        self.btn_delete.clicked.connect(self.delete_selected)
        btn_row.addWidget(self.btn_delete)

        self.btn_clear = QPushButton("清空全部")
        self.btn_clear.setObjectName("Danger")
        self.btn_clear.clicked.connect(self.clear_all)
        btn_row.addWidget(self.btn_clear)
        layout.addLayout(btn_row)

        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels([
            "ID", "时间", "模式", "来源", "时长(秒)", "帧数",
            "峰值人数", "风险告警", "参与度"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self.table)

        detail_group = QFrame()
        detail_group.setObjectName("Card")
        add_shadow(detail_group)
        dl = QVBoxLayout(detail_group)
        dl.setContentsMargins(18, 18, 18, 18)
        dl_title = QLabel("📝 记录详情")
        dl_title.setStyleSheet("font-size:15px;font-weight:bold;color:#1a1a2e;")
        dl.addWidget(dl_title)
        self.lbl_detail = QLabel("选择一条记录查看详情")
        self.lbl_detail.setStyleSheet("font-size:13px;color:#6c7293;")
        self.lbl_detail.setWordWrap(True)
        dl.addWidget(self.lbl_detail)
        layout.addWidget(detail_group)

        self.table.itemSelectionChanged.connect(self.show_detail)

    def load_data(self):
        records = self.db.get_all_records()
        self.table.setRowCount(len(records))
        for i, r in enumerate(records):
            # 新结构: r[3]=scene, r[4]=source, r[5]=duration, r[6]=frames,
            # r[8]=peak_persons, r[11]=risk_alerts, r[10]=participation
            self.table.setItem(i, 0, QTableWidgetItem(str(r[0])))
            self.table.setItem(i, 1, QTableWidgetItem(r[1] or ""))
            self.table.setItem(i, 2, QTableWidgetItem(r[2] or ""))
            self.table.setItem(i, 3, QTableWidgetItem(r[4] or ""))
            self.table.setItem(i, 4, QTableWidgetItem(str(r[5] or 0)))
            self.table.setItem(i, 5, QTableWidgetItem(str(r[6] or 0)))
            self.table.setItem(i, 6, QTableWidgetItem(str(r[8] or 0)))
            self.table.setItem(i, 7, QTableWidgetItem(str(r[11] or 0)))
            self.table.setItem(i, 8, QTableWidgetItem(f"{r[10] or 0:.0%}"))

    def show_detail(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        rid = int(self.table.item(rows[0].row(), 0).text())
        r = self.db.get_record(rid)
        if not r:
            return
        try:
            # 新结构: r[13]=action_dist, r[14]=behavior_dist
            actions = json.loads(r[13]) if r[13] else {}
            behaviors = json.loads(r[14]) if r[14] else {}
        except Exception:
            actions, behaviors = {}, {}

        scene_name = SCENE_CONFIG.get(r[3] if len(r) > 3 else "campus", {}).get("name", r[3] or "")
        detail = (
            f"时间: {r[1]}\n"
            f"模式: {r[2]}  |  场景: {scene_name}  |  来源: {r[4]}\n"
            f"时长: {r[5]}秒  |  帧数: {r[6]}  |  帧率: {r[7]}\n"
            f"峰值人数: {r[8]}  |  平均人数: {r[9]}\n"
            f"参与度: {r[10]:.0%}\n"
            f"风险告警: {r[11]}  |  BlazePose确认: {r[12]}\n"
            f"动作识别: {actions}\n"
            f"行为识别: {behaviors}\n"
            f"结果视频: {r[15] or '无'}"
        )
        self.lbl_detail.setText(detail)

    def delete_selected(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            QMessageBox.information(self, "提示", "请先选择一条记录")
            return
        ret = QMessageBox.question(self, "确认", f"确定删除选中的 {len(rows)} 条记录?")
        if ret == QMessageBox.Yes:
            for row in rows:
                rid = int(self.table.item(row.row(), 0).text())
                self.db.delete_record(rid)
            self.load_data()

    def clear_all(self):
        ret = QMessageBox.question(self, "确认", "确定清空所有历史记录? 此操作不可撤销!")
        if ret == QMessageBox.Yes:
            self.db.clear_all()
            self.load_data()


# ================================================================
# 主窗口
# ================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.db = DatabaseManager()
        self.recent = RecentFilesManager()
        self.current_scene = "campus"
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("智慧体育课堂监测系统 v2.0")
        self.setMinimumSize(1280, 800)
        self.resize(1366, 860)

        central = QWidget()
        central.setObjectName("CentralWidget")
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ===== 侧边栏 =====
        sidebar = QWidget()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(230)
        sb_layout = QVBoxLayout(sidebar)
        sb_layout.setContentsMargins(0, 0, 0, 0)
        sb_layout.setSpacing(0)

        logo = QLabel("⚽  智慧体育课堂")
        logo.setObjectName("LogoLabel")
        sb_layout.addWidget(logo)

        sub = QLabel("质量监测与安全防护平台")
        sub.setObjectName("LogoSub")
        sb_layout.addWidget(sub)

        self.nav_list = QListWidget()
        self.nav_list.setObjectName("NavList")
        nav_items = [
            ("🏠  数据仪表盘", 0),
            ("📷  实时摄像头监测", 1),
            ("🎬  视频文件分析", 2),
            ("📊  课堂统计面板", 3),
            ("📋  历史记录管理", 4),
        ]
        for text, idx in nav_items:
            item = QListWidgetItem(text)
            item.setSizeHint(QSize(210, 52))
            self.nav_list.addItem(item)
        self.nav_list.setCurrentRow(0)
        self.nav_list.currentRowChanged.connect(self.switch_page)
        sb_layout.addWidget(self.nav_list)

        version = QLabel("v2.0  ·  YOLOv8 + BlazePose\n多场景智能监测平台")
        version.setObjectName("VersionLabel")
        version.setAlignment(Qt.AlignCenter)
        sb_layout.addWidget(version)

        main_layout.addWidget(sidebar)

        # ===== 右侧区域 =====
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # 场景选择栏
        scene_bar = QWidget()
        scene_bar.setObjectName("SceneBar")
        scene_bar.setFixedHeight(60)
        sb_layout2 = QHBoxLayout(scene_bar)
        sb_layout2.setContentsMargins(20, 8, 20, 8)
        sb_layout2.setSpacing(8)

        scene_label = QLabel("场景模式:")
        scene_label.setStyleSheet("font-size:13px;font-weight:bold;color:#6c7293;")
        sb_layout2.addWidget(scene_label)

        self.scene_buttons = {}
        for key, cfg in SCENE_CONFIG.items():
            btn = QPushButton(f"{cfg['icon']}  {cfg['name']}")
            btn.setObjectName("SceneBtn")
            btn.setCheckable(True)
            if key == self.current_scene:
                btn.setChecked(True)
            btn.clicked.connect(lambda checked, k=key: self.on_scene_changed(k))
            self.scene_buttons[key] = btn
            sb_layout2.addWidget(btn)

        sb_layout2.addStretch()

        # 当前场景描述
        self.scene_desc_label = QLabel(SCENE_CONFIG[self.current_scene]["desc"])
        self.scene_desc_label.setStyleSheet("font-size:12px;color:#8b8ba0;")
        sb_layout2.addWidget(self.scene_desc_label)

        right_layout.addWidget(scene_bar)

        # 内容区
        content = QWidget()
        content.setObjectName("Content")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(24, 20, 24, 20)
        content_layout.setSpacing(16)

        self.stack = QStackedWidget()
        self.dashboard_page = DashboardPage(
            self.db, self.recent, self.on_scene_changed_from_dashboard)
        self.camera_page = CameraPage(self.db, self.current_scene)
        self.video_page = VideoAnalyzePage(self.db, self.recent, self.current_scene)
        self.stats_page = StatsPanelPage(self.db)
        self.history_page = HistoryPage(self.db)
        self.stack.addWidget(self.dashboard_page)
        self.stack.addWidget(self.camera_page)
        self.stack.addWidget(self.video_page)
        self.stack.addWidget(self.stats_page)
        self.stack.addWidget(self.history_page)
        content_layout.addWidget(self.stack)

        right_layout.addWidget(content)
        main_layout.addWidget(right)

    def on_scene_changed(self, key):
        self.current_scene = key
        for k, btn in self.scene_buttons.items():
            btn.setChecked(k == key)
        self.scene_desc_label.setText(SCENE_CONFIG[key]["desc"])
        # 更新各页面文案
        self.camera_page.update_scene(key)
        self.video_page.update_scene(key)
        self.stats_page.update_scene(key)

    def on_scene_changed_from_dashboard(self, key):
        self.on_scene_changed(key)

    def switch_page(self, idx):
        self.stack.setCurrentIndex(idx)
        if idx == 0:
            self.dashboard_page.load_data()
        elif idx == 2:
            self.video_page.refresh_recent()
        elif idx == 3:
            self.stats_page.load_data()
        elif idx == 4:
            self.history_page.load_data()

    def closeEvent(self, event):
        if self.camera_page.camera_thread:
            self.camera_page.camera_thread.stop()
            self.camera_page.camera_thread.wait()
        if self.video_page.process_thread:
            self.video_page.process_thread.stop()
            self.video_page.process_thread.wait()
        event.accept()


# ================================================================
# 入口
# ================================================================
def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE_SHEET)
    app.setFont(QFont("Microsoft YaHei", 10))

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()