# -*- coding: utf-8 -*-
"""
校园智慧体育课堂 - Web 服务端
================================
前后端分离架构：手机/电脑浏览器均可访问
  - 后端：FastAPI 封装 YOLOv8 + BlazePose 检测引擎
  - 前端：响应式 H5 页面（static/index.html）

启动：
  python web_server.py
  python web_server.py --port 8080
手机访问：同一局域网下浏览器打开 http://<电脑IP>:8000
"""

import argparse
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import threading
import time
import uuid
from collections import Counter
from datetime import datetime

import cv2
import numpy as np
import sqlite3
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, WebSocket, WebSocketDisconnect, Request, Depends
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import base64

# ===== 导入检测引擎 =====
import torch  # noqa: F401
from ultralytics import YOLO  # noqa: F401
from sports_classroom_system import SportsClassroomSystem, draw_panel

# ===== 路径配置 =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Web 端独立数据库，与桌面客户端数据完全隔离
DB_PATH = os.path.join(BASE_DIR, "web_history.db")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "results")
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)


# ================================================================
# 视频编码工具：用 ffmpeg 将 OpenCV 输出的 mp4v 转为 H.264（浏览器可播放）
# ================================================================
def get_ffmpeg_path():
    """查找可用的 ffmpeg 可执行文件路径"""
    # 1. 系统 PATH
    p = shutil.which("ffmpeg")
    if p:
        return p
    # 2. TRAE 自带的 ffmpeg
    candidates = [
        os.path.expandvars(
            r"%LOCALAPPDATA%\Programs\TRAE SOLO CN\resources\app\bin\ffmpeg.exe"),
        r"C:\Users\pc\AppData\Local\Programs\TRAE SOLO CN\resources\app\bin\ffmpeg.exe",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def convert_to_h264(input_path):
    """用 ffmpeg 将视频重新编码为 H.264 + yuv420p（浏览器兼容），原地替换。"""
    ffmpeg = get_ffmpeg_path()
    if not ffmpeg:
        print("[视频转换] 未找到 ffmpeg，保留原始编码")
        return False

    tmp_path = input_path + ".h264_tmp.mp4"
    cmd = [
        ffmpeg, "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",        # 浏览器兼容的像素格式
        "-movflags", "+faststart",     # 支持流式播放（moov 在前）
        "-an",                         # 无音频
        tmp_path,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, timeout=600,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode == 0 and os.path.exists(tmp_path):
            os.remove(input_path)
            os.rename(tmp_path, input_path)
            print(f"[视频转换] H.264 编码完成: {os.path.basename(input_path)}")
            return True
        else:
            stderr = result.stderr.decode("utf-8", errors="replace")[-500:]
            print(f"[视频转换] ffmpeg 失败 (code={result.returncode}): {stderr}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return False
    except Exception as e:
        print(f"[视频转换] 异常: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False


# ================================================================
# SSL 证书生成（用于 HTTPS，让浏览器允许摄像头访问）
# ================================================================
def generate_self_signed_cert():
    """生成自签名 SSL 证书，返回 (cert_path, key_path) 或 (None, None)"""
    cert_dir = os.path.join(BASE_DIR, "ssl")
    os.makedirs(cert_dir, exist_ok=True)
    cert_file = os.path.join(cert_dir, "cert.pem")
    key_file = os.path.join(cert_dir, "key.pem")

    if os.path.exists(cert_file) and os.path.exists(key_file):
        return cert_file, key_file

    # 方式1: 使用 cryptography 库
    try:
        import datetime as _dt
        import ipaddress
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "CN"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "SportsClassroom"),
            x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
        ])
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(_dt.datetime.utcnow())
            .not_valid_after(_dt.datetime.utcnow() + _dt.timedelta(days=365))
            .add_extension(
                x509.SubjectAlternativeName([
                    x509.DNSName("localhost"),
                    x509.DNSName("*"),
                    x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
                ]),
                critical=False,
            )
            .sign(key, hashes.SHA256())
        )
        with open(cert_file, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        with open(key_file, "wb") as f:
            f.write(key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            ))
        print("[SSL] 自签名证书已生成 (cryptography)")
        return cert_file, key_file
    except ImportError:
        pass
    except Exception as e:
        print(f"[SSL] cryptography 生成失败: {e}")

    # 方式2: 使用 openssl 命令
    try:
        subprocess.run([
            "openssl", "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", key_file, "-out", cert_file,
            "-days", "365", "-nodes", "-subj", "/CN=localhost",
        ], check=True, capture_output=True, timeout=30)
        print("[SSL] 自签名证书已生成 (openssl)")
        return cert_file, key_file
    except Exception:
        pass

    return None, None

# ===== 场景配置 =====
SCENE_CONFIG = {
    "campus": {
        "name": "校园体育", "icon": "🎓",
        "color": "#4facfe", "color2": "#00f2fe",
        "desc": "课堂质量监测 · 学生安全防护",
        "features": ["课堂参与度", "运动安全告警", "行为识别", "到课统计"],
        "count_label": "到课人数", "part_label": "课堂参与度",
        "state_label": "课堂状态",
    },
    "fitness": {
        "name": "健身训练", "icon": "💪",
        "color": "#ff9f43", "color2": "#ff7847",
        "desc": "动作标准分析 · 训练效果评估",
        "features": ["动作识别", "姿势纠正", "训练计数", "强度评估"],
        "count_label": "训练人数", "part_label": "训练活跃度",
        "state_label": "训练状态",
    },
    "rehab": {
        "name": "康复训练", "icon": "🏥",
        "color": "#00d8a0", "color2": "#00b894",
        "desc": "活动范围监测 · 恢复进度追踪",
        "features": ["关节活动度", "动作范围", "恢复评估", "温和提示"],
        "count_label": "训练人数", "part_label": "活动完成度",
        "state_label": "康复状态",
    },
    "general": {
        "name": "通用模式", "icon": "⚡",
        "color": "#a855f7", "color2": "#7c3aed",
        "desc": "全功能检测 · 多场景适配",
        "features": ["全部检测项", "自定义配置", "灵活应用", "数据导出"],
        "count_label": "检测人数", "part_label": "活动参与度",
        "state_label": "检测状态",
    },
}


# ================================================================
# 全局检测引擎（实时分析复用，避免重复加载模型）
# ================================================================
_global_engine = None
_global_engine_lock = threading.Lock()
_global_engine_blazepose = None
_global_engine_scene = None


def get_engine(use_blazepose=True, scene="campus"):
    """获取全局检测引擎实例（线程安全），支持场景切换"""
    global _global_engine, _global_engine_blazepose, _global_engine_scene
    with _global_engine_lock:
        if _global_engine is None or _global_engine_blazepose != use_blazepose:
            _global_engine = SportsClassroomSystem(use_blazepose=use_blazepose, scene=scene)
            _global_engine_blazepose = use_blazepose
            _global_engine_scene = scene
        elif _global_engine_scene != scene:
            # 场景变了，动态切换（无需重新加载模型）
            _global_engine.set_scene(scene)
            _global_engine_scene = scene
        return _global_engine


# ================================================================
# 数据库管理（Web 端独立数据库，与桌面端隔离）
# ================================================================
class DatabaseManager:
    def __init__(self, db_path=None):
        self.db_path = db_path or DB_PATH
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        # 用户表
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                display_name TEXT,
                created_at TEXT NOT NULL
            )
        """)
        # 登录令牌表
        c.execute("""
            CREATE TABLE IF NOT EXISTS tokens (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        # 监测记录表
        c.execute("""
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER DEFAULT 0,
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
        # 迁移：旧表无 user_id 列时自动添加
        try:
            c.execute("SELECT user_id FROM records LIMIT 1")
        except sqlite3.OperationalError:
            c.execute("ALTER TABLE records ADD COLUMN user_id INTEGER DEFAULT 0")
        conn.commit()
        conn.close()

    # ===== 用户管理 =====
    def create_user(self, username, password, display_name=None):
        salt = secrets.token_hex(16)
        pw_hash = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000
        ).hex()
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        try:
            c.execute(
                "INSERT INTO users (username, password_hash, salt, display_name, created_at) "
                "VALUES (?,?,?,?,?)",
                (username, pw_hash, salt, display_name or username,
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            )
            conn.commit()
            uid = c.lastrowid
            conn.close()
            return uid, None
        except sqlite3.IntegrityError:
            conn.close()
            return None, "用户名已存在"

    def verify_user(self, username, password):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT id, password_hash, salt, display_name FROM users WHERE username=?",
                  (username,))
        row = c.fetchone()
        conn.close()
        if not row:
            return None, "用户不存在"
        uid, pw_hash, salt, display = row
        calc = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000
        ).hex()
        if calc != pw_hash:
            return None, "密码错误"
        return {"id": uid, "username": username, "display_name": display}, None

    def create_token(self, user_id):
        token = secrets.token_urlsafe(32)
        now = datetime.now()
        expires = datetime(now.year + 1, now.month, now.day)
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            "INSERT INTO tokens (token, user_id, created_at, expires_at) VALUES (?,?,?,?)",
            (token, user_id, now.strftime("%Y-%m-%d %H:%M:%S"),
             expires.strftime("%Y-%m-%d %H:%M:%S")),
        )
        conn.commit()
        conn.close()
        return token

    def get_user_by_token(self, token):
        if not token:
            return None
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            "SELECT u.id, u.username, u.display_name FROM tokens t "
            "JOIN users u ON t.user_id = u.id "
            "WHERE t.token = ? AND t.expires_at > ?",
            (token, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
        row = c.fetchone()
        conn.close()
        if not row:
            return None
        return {"id": row[0], "username": row[1], "display_name": row[2]}

    def delete_token(self, token):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("DELETE FROM tokens WHERE token=?", (token,))
        conn.commit()
        conn.close()

    # ===== 记录管理（带用户过滤）=====
    def save_record(self, data):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            INSERT INTO records (user_id, timestamp, mode, scene, source, duration_sec,
                total_frames, avg_fps, peak_persons, avg_persons,
                participation, risk_alerts, blaze_confirm,
                action_dist, behavior_dist, report_path)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            data.get("user_id", 0),
            data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            data.get("mode", ""), data.get("scene", "campus"),
            data.get("source", ""), data.get("duration_sec", 0),
            data.get("total_frames", 0), data.get("avg_fps", 0),
            data.get("peak_persons", 0), data.get("avg_persons", 0),
            data.get("participation", 0), data.get("risk_alerts", 0),
            data.get("blaze_confirm", 0),
            json.dumps(data.get("action_dist", {}), ensure_ascii=False),
            json.dumps(data.get("behavior_dist", {}), ensure_ascii=False),
            data.get("report_path", ""),
        ))
        conn.commit()
        rid = c.lastrowid
        conn.close()
        return rid

    def get_all_records(self, user_id, limit=50):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT * FROM records WHERE user_id=? ORDER BY id DESC LIMIT ?",
                  (user_id, limit))
        rows = c.fetchall()
        conn.close()
        return rows

    def get_record(self, rid, user_id):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT * FROM records WHERE id=? AND user_id=?", (rid, user_id))
        row = c.fetchone()
        conn.close()
        return row

    def delete_record(self, rid, user_id):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("DELETE FROM records WHERE id=? AND user_id=?", (rid, user_id))
        conn.commit()
        conn.close()

    def get_stats(self, user_id):
        """获取汇总统计（按用户过滤），包含总览和分场景统计"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        # 总览
        c.execute("SELECT COUNT(*), COALESCE(SUM(total_frames),0), "
                  "COALESCE(SUM(risk_alerts),0), COALESCE(AVG(participation),0) "
                  "FROM records WHERE user_id=?", (user_id,))
        row = c.fetchone()
        overall = {
            "total_sessions": row[0] or 0,
            "total_frames": row[1] or 0,
            "total_alerts": row[2] or 0,
            "avg_participation": round(min(1.0, max(0.0, row[3] or 0)), 3),
        }
        # 分场景统计
        c.execute(
            "SELECT scene, COUNT(*), COALESCE(SUM(total_frames),0), "
            "COALESCE(SUM(risk_alerts),0), COALESCE(AVG(participation),0), "
            "COALESCE(AVG(avg_persons),0), COALESCE(MAX(peak_persons),0) "
            "FROM records WHERE user_id=? GROUP BY scene",
            (user_id,))
        scenes = {}
        for srow in c.fetchall():
            scenes[srow[0]] = {
                "sessions": srow[1] or 0,
                "total_frames": srow[2] or 0,
                "total_alerts": srow[3] or 0,
                "avg_participation": round(min(1.0, max(0.0, srow[4] or 0)), 3),
                "avg_persons": round(srow[5] or 0, 1),
                "peak_persons": srow[6] or 0,
            }
        conn.close()
        return {**overall, "scenes": scenes}


db = DatabaseManager()


# ================================================================
# 认证依赖
# ================================================================
def get_token_from_request(request: Request):
    """从请求中提取 token（header 或 query 参数）"""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return request.query_params.get("token", "")


def require_user(request: Request):
    """依赖项：要求已登录，返回用户信息"""
    token = get_token_from_request(request)
    user = db.get_user_by_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="未登录或登录已过期")
    return user


# ================================================================
# 异步任务管理器
# ================================================================
class TaskManager:
    """管理异步视频分析任务"""

    def __init__(self):
        self.tasks = {}
        self.lock = threading.Lock()
        self._engine_lock = threading.Lock()  # 检测引擎互斥锁

    def create_task(self, video_path, output_path, use_blazepose, scene, source_name, user_id):
        task_id = uuid.uuid4().hex[:8]
        task_info = {
            "task_id": task_id,
            "status": "pending",
            "progress": 0,
            "total_frames": 0,
            "processed_frames": 0,
            "result": None,
            "error": None,
            "created_at": time.time(),
            "source_name": source_name,
            "scene": scene,
            "video_path": video_path,
            "output_path": output_path,
            "use_blazepose": use_blazepose,
            "user_id": user_id,
        }
        with self.lock:
            self.tasks[task_id] = task_info
        thread = threading.Thread(target=self._process, args=(task_id,), daemon=True)
        thread.start()
        return task_id

    def _process(self, task_id):
        task = self.tasks[task_id]
        try:
            task["status"] = "processing"
            # 检测引擎互斥（避免并发加载模型冲突），传入场景模式
            with self._engine_lock:
                system = SportsClassroomSystem(
                    use_blazepose=task["use_blazepose"],
                    scene=task.get("scene", "campus"))

            cap = cv2.VideoCapture(task["video_path"])
            if not cap.isOpened():
                task["status"] = "failed"
                task["error"] = "无法打开视频文件"
                return

            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_src = cap.get(cv2.CAP_PROP_FPS) or 25
            task["total_frames"] = total

            w_src = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            h_src = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
            # 优先 H.264 编码（浏览器可播放），失败回退 mp4v
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            writer = cv2.VideoWriter(task["output_path"], fourcc, fps_src, (w_src, h_src))
            if not writer.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*"H264")
                writer = cv2.VideoWriter(task["output_path"], fourcc, fps_src, (w_src, h_src))
            if not writer.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(task["output_path"], fourcc, fps_src, (w_src, h_src))

            total_frames = 0
            sum_total = sum_moving = sum_standing = sum_still = 0
            sum_alerts = sum_confirmed = 0
            max_total = 0
            action_dist = Counter()
            behavior_dist = Counter()
            prev_t = time.time()
            fps_smooth = 0.0
            start_time = time.time()

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

                task["processed_frames"] = total_frames
                if total > 0:
                    task["progress"] = int(total_frames / total * 100)

            duration = time.time() - start_time
            cap.release()
            if writer:
                writer.release()
            system.close()

            # ===== 用 ffmpeg 转为 H.264 编码（浏览器可播放）=====
            if os.path.exists(task["output_path"]):
                task["status"] = "encoding"  # 告知前端正在编码
                convert_to_h264(task["output_path"])

            participation = round(
                min(1.0, max(0.0, (sum_moving + sum_standing) / max(sum_total, 1))), 3)

            report = {
                "total_frames": total_frames,
                "duration_sec": round(duration, 1),
                "avg_fps": round(fps_smooth, 1),
                "peak_persons": max_total,
                "avg_persons": round(sum_total / max(total_frames, 1), 1),
                "participation": participation,
                "risk_alerts": sum_alerts,
                "blaze_confirm": sum_confirmed,
                "action_dist": dict(action_dist),
                "behavior_dist": dict(behavior_dist),
                "output_path": task["output_path"],
                "moving_frames": sum_moving,
                "standing_frames": sum_standing,
                "still_frames": sum_still,
            }

            # 保存到数据库
            rid = db.save_record({
                "user_id": task["user_id"],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "mode": "Web视频分析",
                "scene": task["scene"],
                "source": task["source_name"],
                "duration_sec": report["duration_sec"],
                "total_frames": report["total_frames"],
                "avg_fps": report["avg_fps"],
                "peak_persons": report["peak_persons"],
                "avg_persons": report["avg_persons"],
                "participation": report["participation"],
                "risk_alerts": report["risk_alerts"],
                "blaze_confirm": report["blaze_confirm"],
                "action_dist": report["action_dist"],
                "behavior_dist": report["behavior_dist"],
                "report_path": task["output_path"],
            })
            report["record_id"] = rid

            task["result"] = report
            task["status"] = "completed"
            task["progress"] = 100

        except Exception as e:
            task["status"] = "failed"
            task["error"] = str(e)

    def get_task(self, task_id):
        with self.lock:
            return self.tasks.get(task_id)


task_manager = TaskManager()


# ================================================================
# FastAPI 应用
# ================================================================
app = FastAPI(title="校园智慧体育课堂监测平台", version="1.0")

# 全局公网地址（启动时设置）
PUBLIC_URL = None

# 静态文件
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _row_to_dict(row):
    """数据库行转字典（含 user_id 列）"""
    if not row:
        return None
    def safe_json(val):
        if not val:
            return {}
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return {}
    return {
        "id": row[0], "user_id": row[1], "timestamp": row[2], "mode": row[3],
        "scene": row[4], "source": row[5], "duration_sec": row[6],
        "total_frames": row[7], "avg_fps": row[8], "peak_persons": row[9],
        "avg_persons": row[10], "participation": row[11], "risk_alerts": row[12],
        "blaze_confirm": row[13],
        "action_dist": safe_json(row[14]),
        "behavior_dist": safe_json(row[15]),
        "report_path": row[16] or "",
    }


# ===== 路由 =====

# ===== 认证 API =====
class AuthRequest(BaseModel):
    username: str
    password: str
    display_name: str = ""


@app.post("/api/auth/register")
async def register(req: AuthRequest):
    """用户注册"""
    username = req.username.strip()
    if len(username) < 2:
        raise HTTPException(status_code=400, detail="用户名至少2个字符")
    if len(req.password) < 4:
        raise HTTPException(status_code=400, detail="密码至少4个字符")
    uid, err = db.create_user(username, req.password, req.display_name or None)
    if err:
        raise HTTPException(status_code=400, detail=err)
    token = db.create_token(uid)
    user = db.get_user_by_token(token)
    return JSONResponse(content={"token": token, "user": user, "message": "注册成功"})


@app.post("/api/auth/login")
async def login(req: AuthRequest):
    """用户登录"""
    user, err = db.verify_user(req.username.strip(), req.password)
    if err:
        raise HTTPException(status_code=400, detail=err)
    token = db.create_token(user["id"])
    return JSONResponse(content={"token": token, "user": user, "message": "登录成功"})


@app.post("/api/auth/logout")
async def logout(request: Request):
    """退出登录"""
    token = get_token_from_request(request)
    if token:
        db.delete_token(token)
    return JSONResponse(content={"message": "已退出"})


@app.get("/api/auth/me")
async def get_current_user(request: Request):
    """获取当前登录用户"""
    token = get_token_from_request(request)
    user = db.get_user_by_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="未登录")
    return JSONResponse(content={"user": user})


@app.get("/", response_class=HTMLResponse)
async def index():
    """返回前端页面（禁缓存，确保每次拿到最新版本）"""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(
            content=content,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    return HTMLResponse(content="<h1>前端页面未找到，请检查 static/index.html</h1>")


@app.get("/api/scenes")
async def get_scenes():
    """获取场景配置"""
    return JSONResponse(content=SCENE_CONFIG)


@app.get("/api/public-url")
async def get_public_url():
    """获取公网访问地址"""
    return JSONResponse(content={"public_url": PUBLIC_URL, "has_public": PUBLIC_URL is not None})


@app.get("/api/qrcode")
async def get_qrcode():
    """返回公网地址二维码图片"""
    if not PUBLIC_URL:
        raise HTTPException(status_code=404, detail="公网未启用")
    qr_path = os.path.join(STATIC_DIR, "qrcode.png")
    if not os.path.exists(qr_path):
        raise HTTPException(status_code=404, detail="二维码未生成")
    return FileResponse(qr_path, media_type="image/png")


@app.get("/api/stats")
async def get_stats(user: dict = Depends(require_user)):
    """获取汇总统计（按用户过滤）"""
    return JSONResponse(content=db.get_stats(user["id"]))


@app.post("/api/analyze")
async def analyze_video(
    request: Request,
    file: UploadFile = File(...),
    use_blazepose: bool = Query(True),
    scene: str = Query("campus"),
):
    """上传视频并启动分析任务（需登录，文件按用户隔离存储）"""
    token = get_token_from_request(request)
    user = db.get_user_by_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="未登录或登录已过期")

    if not file.filename:
        raise HTTPException(status_code=400, detail="未提供文件")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".mp4", ".avi", ".mov", ".mkv", ".flv"):
        raise HTTPException(status_code=400, detail=f"不支持的视频格式: {ext}")

    # 用户专属存储目录
    user_upload = os.path.join(UPLOAD_DIR, str(user["id"]))
    user_result = os.path.join(RESULT_DIR, str(user["id"]))
    os.makedirs(user_upload, exist_ok=True)
    os.makedirs(user_result, exist_ok=True)

    # 保存上传文件
    save_name = f"upload_{int(time.time())}_{uuid.uuid4().hex[:6]}{ext}"
    video_path = os.path.join(user_upload, save_name)
    content = await file.read()
    with open(video_path, "wb") as f:
        f.write(content)

    # 结果视频路径
    out_name = f"result_{save_name}"
    output_path = os.path.join(user_result, out_name)

    # 创建分析任务
    task_id = task_manager.create_task(
        video_path, output_path, use_blazepose, scene, file.filename, user["id"])

    return JSONResponse(content={
        "task_id": task_id,
        "message": "视频已上传，分析任务已启动",
        "filename": file.filename,
    })


@app.get("/api/task/{task_id}")
async def get_task_status(task_id: str, user: dict = Depends(require_user)):
    """查询任务状态"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    if task.get("user_id") != user["id"]:
        raise HTTPException(status_code=403, detail="无权访问此任务")
    return JSONResponse(content={
        "task_id": task["task_id"],
        "status": task["status"],
        "progress": task["progress"],
        "total_frames": task["total_frames"],
        "processed_frames": task["processed_frames"],
        "result": task["result"],
        "error": task["error"],
        "source_name": task["source_name"],
    })


@app.get("/api/task/{task_id}/video")
async def download_result_video(task_id: str, user: dict = Depends(require_user)):
    """下载结果视频"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    if task.get("user_id") != user["id"]:
        raise HTTPException(status_code=403, detail="无权访问此任务")
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="任务尚未完成")
    output_path = task["output_path"]
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="结果视频文件不存在")
    return FileResponse(output_path, media_type="video/mp4",
                        filename=os.path.basename(output_path))


@app.get("/api/records")
async def get_records(limit: int = Query(50, le=200), user: dict = Depends(require_user)):
    """获取历史记录列表（按用户过滤）"""
    rows = db.get_all_records(user["id"], limit)
    records = []
    for row in rows:
        d = _row_to_dict(row)
        records.append({
            "id": d["id"], "timestamp": d["timestamp"],
            "mode": d["mode"], "scene": d["scene"],
            "source": d["source"], "duration_sec": d["duration_sec"],
            "total_frames": d["total_frames"],
            "peak_persons": d["peak_persons"],
            "participation": d["participation"],
            "risk_alerts": d["risk_alerts"],
        })
    return JSONResponse(content={"records": records, "count": len(records)})


@app.get("/api/records/{rid}")
async def get_record_detail(rid: int, user: dict = Depends(require_user)):
    """获取单条记录详情（按用户过滤）"""
    row = db.get_record(rid, user["id"])
    if not row:
        raise HTTPException(status_code=404, detail="记录不存在")
    return JSONResponse(content=_row_to_dict(row))


@app.delete("/api/records/{rid}")
async def delete_record(rid: int, user: dict = Depends(require_user)):
    """删除记录（按用户过滤）"""
    db.delete_record(rid, user["id"])
    return JSONResponse(content={"message": "已删除"})


# ===== WebSocket 实时分析 =====
@app.websocket("/ws/realtime")
async def websocket_realtime(ws: WebSocket):
    """WebSocket 实时摄像头分析（需登录，通过 query 参数 token 认证）
    前端流程: getUserMedia → canvas截图 → base64发送 → 接收标注帧+统计
    消息格式(入): {"type":"frame","data":"base64jpg","use_bp":true}
    消息格式(出): {"type":"result","frame":"base64jpg","stats":{...}}
    """
    # 从 query 参数获取 token 验证用户
    token = ws.query_params.get("token", "")
    user = db.get_user_by_token(token)
    if not user:
        await ws.accept()
        await ws.send_text(json.dumps({"type": "error", "message": "未登录或登录已过期，请重新登录"}))
        await ws.close(code=1008)
        return

    await ws.accept()
    print(f"[WebSocket] 用户 {user['username']} 已连接实时监测")
    engine = None
    prev_t = time.time()
    fps_smooth = 0.0
    # 累计统计
    sum_total = sum_moving = sum_standing = sum_still = 0
    sum_alerts = 0
    peak_persons = 0
    frame_count = 0
    start_time = time.time()
    cur_scene = "campus"

    try:
        while True:
            msg = await ws.receive_text()
            try:
                data = json.loads(msg)
            except Exception:
                continue

            if data.get("type") != "frame":
                continue

            cur_scene = data.get("scene", "campus")

            # 首帧时初始化引擎，后续帧切换场景
            if engine is None:
                use_bp = data.get("use_bp", True)
                engine = get_engine(use_bp, cur_scene)
            else:
                # 场景变化时动态切换
                engine = get_engine(data.get("use_bp", True), cur_scene)

            # 解码 base64 图片
            img_b64 = data.get("data", "")
            if "," in img_b64:
                img_b64 = img_b64.split(",", 1)[1]
            try:
                img_bytes = base64.b64decode(img_b64)
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
            except Exception:
                continue

            # 分析帧
            frame, stats = engine.process_frame(frame)
            now = time.time()
            inst_fps = 1.0 / max(now - prev_t, 1e-6)
            prev_t = now
            fps_smooth = 0.8 * fps_smooth + 0.2 * inst_fps if fps_smooth else inst_fps
            draw_panel(frame, stats, fps_smooth, engine.use_blazepose, engine.scene)

            # 累计统计
            frame_count += 1
            sum_total += stats["total"]
            sum_moving += stats["moving"]
            sum_standing += stats["standing"]
            sum_still += stats["still"]
            sum_alerts += stats["alerts"]
            if stats["total"] > peak_persons:
                peak_persons = stats["total"]

            # 编码返回（降低质量和尺寸以提升传输速度）
            h, w = frame.shape[:2]
            if max(h, w) > 640:
                scale = 640.0 / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            _, buf = cv2.imencode(".jpg", frame,
                                  [cv2.IMWRITE_JPEG_QUALITY, 60])
            frame_b64 = base64.b64encode(buf).decode("utf-8")

            participation = min(1.0, max(0.0, (stats["moving"] + stats["standing"]) / max(stats["total"], 1))) if stats["total"] > 0 else 0

            result = {
                "type": "result",
                "frame": "data:image/jpeg;base64," + frame_b64,
                "stats": {
                    "total": stats["total"],
                    "moving": stats["moving"],
                    "standing": stats["standing"],
                    "still": stats["still"],
                    "alerts": stats["alerts"],
                    "cur_actions": stats.get("cur_actions", []),
                    "cur_behaviors": stats.get("cur_behaviors", []),
                    "fps": round(fps_smooth, 1),
                    "participation": round(participation, 3),
                    "scene": engine.scene,
                    "scene_metrics": stats.get("scene_metrics", {}),
                },
                "cumulative": {
                    "frame_count": frame_count,
                    "duration": round(time.time() - start_time, 1),
                    "peak_persons": peak_persons,
                    "sum_alerts": sum_alerts,
                    "avg_participation": round(
                        min(1.0, max(0.0, (sum_moving + sum_standing) / max(sum_total, 1))), 3),
                },
            }
            await ws.send_text(json.dumps(result))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WebSocket] 异常: {e}")
    finally:
        # 保存实时监测记录
        if frame_count > 10:
            duration = time.time() - start_time
            participation = round(
                min(1.0, max(0.0, (sum_moving + sum_standing) / max(sum_total, 1))), 3)
            db.save_record({
                "user_id": user["id"],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "mode": "Web实时监测",
                "scene": cur_scene,
                "source": "摄像头实时",
                "duration_sec": round(duration, 1),
                "total_frames": frame_count,
                "avg_fps": round(fps_smooth, 1),
                "peak_persons": peak_persons,
                "avg_persons": round(sum_total / max(frame_count, 1), 1),
                "participation": participation,
                "risk_alerts": sum_alerts,
                "blaze_confirm": 0,
                "action_dist": {},
                "behavior_dist": {},
                "report_path": "",
            })


# ===== 公网穿透 =====
def start_ngrok(port):
    """通过 pyngrok 启动公网隧道，返回公网 URL。
    需要先注册 ngrok 免费账号获取 authtoken: https://dashboard.ngrok.com/signup
    然后运行: ngrok config add-authtoken <你的token>
    或设置环境变量 NGROK_AUTHTOKEN
    """
    try:
        from pyngrok import ngrok, conf
        # 读取环境变量中的 authtoken
        authtoken = os.environ.get("NGROK_AUTHTOKEN", "")
        if authtoken:
            conf.get_default().auth_token = authtoken
        # 启动隧道
        public_url = ngrok.connect(port, "http").public_url
        return public_url, None
    except ImportError:
        return None, "pyngrok 未安装，请运行: pip install pyngrok"
    except Exception as e:
        err_msg = str(e)
        if "authtoken" in err_msg.lower():
            return None, ("ngrok 需要 authtoken 认证。\n"
                "1. 注册免费账号: https://dashboard.ngrok.com/signup\n"
                "2. 获取 authtoken 后运行:\n"
                "   set NGROK_AUTHTOKEN=你的token\n"
                "3. 重新启动本服务")
        return None, f"ngrok 启动失败: {err_msg}"


def start_cloudflared(port, https=False):
    """通过 cloudflared Quick Tunnel 启动公网隧道，无需注册。
    安装: winget install --id Cloudflare.cloudflared
    """
    import subprocess
    import shutil
    try:
        # 检查 cloudflared 是否可用（PATH + 常见安装路径）
        cf_path = shutil.which("cloudflared")
        if not cf_path:
            # winget 安装后可能不在 PATH，查找常见路径
            candidates = [
                r"C:\Program Files (x86)\cloudflared\cloudflared.exe",
                r"C:\Program Files\cloudflared\cloudflared.exe",
                os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Cloudflare.cloudflared_Microsoft.Winget.Source_8wekyb3d8bbwe\cloudflared.exe"),
            ]
            for p in candidates:
                if os.path.isfile(p):
                    cf_path = p
                    break
        if not cf_path:
            return None, ("cloudflared 未安装。\n"
                "安装命令: winget install --id Cloudflare.cloudflared")
        # 后端协议：HTTPS 服务器用 https:// 连接，否则 http://
        backend_scheme = "https" if https else "http"
        cmd = [cf_path, "tunnel", "--url", f"{backend_scheme}://127.0.0.1:{port}",
               "--no-autoupdate"]
        # HTTPS 后端是自签名证书，需要跳过证书验证
        if https:
            cmd += ["--no-tls-verify"]
        # 启动 Quick Tunnel（无需域名/账号）
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace")
        # 等待输出中包含 trycloudflare.com 的 URL
        import time as _time
        import re
        start = _time.time()
        while _time.time() - start < 45:
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                continue
            line = line.strip()
            # cloudflared 日志可能在 stderr 输出 URL
            match = re.search(r'https://[a-z0-9-]+\.trycloudflare\.com', line)
            if match:
                return match.group(0), None
        return None, "cloudflared 45秒内未获取到公网地址"
    except FileNotFoundError:
        return None, "cloudflared 未安装"
    except Exception as e:
        return None, f"cloudflared 启动失败: {str(e)}"


# ===== 启动 =====
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="校园智慧体育课堂 Web 服务")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--https", action="store_true",
                        help="启用 HTTPS（自签名证书），手机/电脑通过IP访问时摄像头需要HTTPS")
    parser.add_argument("--tunnel", choices=["ngrok", "cloudflare", "off"],
                        default="off",
                        help="公网穿透模式: ngrok(需token) / cloudflare(免注册) / off(仅局域网)")
    args = parser.parse_args()

    print("=" * 60)
    print("  校园智慧体育课堂监测平台 - Web 服务")
    print("=" * 60)

    # 获取本机 IP
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "127.0.0.1"

    print(f"\n  局域网访问:")
    protocol = "https" if args.https else "http"
    print(f"    本机:   {protocol}://127.0.0.1:{args.port}")
    print(f"    手机:   {protocol}://{local_ip}:{args.port}")
    print(f"    API文档: {protocol}://127.0.0.1:{args.port}/docs")
    if not args.https:
        print(f"\n  ⚠️  当前为 HTTP 模式：通过IP访问时摄像头将无法使用")
        print(f"     如需摄像头功能，请加 --https 参数启动")
        print(f"     或仅通过 http://127.0.0.1:{args.port} 本机访问")

    # 公网穿透
    public_url = None
    if args.tunnel == "ngrok":
        print(f"\n  正在启动 ngrok 公网隧道...")
        public_url, err = start_ngrok(args.port)
        if public_url:
            print(f"  ✅ 公网访问: {public_url}")
            print(f"     任何设备均可访问，不受网络限制")
        else:
            print(f"  ❌ {err}")

    elif args.tunnel == "cloudflare":
        print(f"\n  正在启动 Cloudflare Tunnel (免注册)...")
        public_url, err = start_cloudflared(args.port, https=args.https)
        if public_url:
            print(f"  ✅ 公网访问: {public_url}")
            print(f"     任何设备均可访问，不受网络限制")
        else:
            print(f"  ❌ {err}")
            print(f"     安装: winget install --id Cloudflare.cloudflared")

    if not public_url and args.tunnel == "off":
        print(f"\n  提示: 添加 --tunnel cloudflare 可启用公网访问(免注册)")
        print(f"        添加 --tunnel ngrok 可启用公网访问(需token)")

    # 设置全局公网地址 + 生成二维码
    if public_url:
        PUBLIC_URL = public_url
        try:
            import qrcode
            qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L,
                               box_size=8, border=2)
            qr.add_data(public_url)
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white")
            qr_path = os.path.join(STATIC_DIR, "qrcode.png")
            img.save(qr_path)
            print(f"  📱 二维码已生成，手机扫码即可访问")
        except Exception:
            pass  # 二维码生成失败不影响服务

    print(f"\n  按 Ctrl+C 停止服务")
    print("=" * 60 + "\n")

    # ===== 启动服务（可选 HTTPS）=====
    if args.https:
        cert_file, key_file = generate_self_signed_cert()
        if cert_file and key_file:
            print(f"  🔒 HTTPS 已启用（自签名证书）")
            print(f"     首次访问浏览器会提示不安全，点击「高级」→「继续」即可")
            print(f"     摄像头功能需要 HTTPS 或 localhost 才能使用\n")
            uvicorn.run(app, host=args.host, port=args.port,
                        log_level="info",
                        ssl_certfile=cert_file, ssl_keyfile=key_file)
        else:
            print(f"  ❌ SSL 证书生成失败，回退到 HTTP 模式")
            print(f"     提示: pip install cryptography 可启用 HTTPS\n")
            uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    else:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
