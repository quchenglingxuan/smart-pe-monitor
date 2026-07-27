# 校园智慧体育课堂质量监测与安全防护服务平台

> 基于 YOLOv8 + BlazePose 的智能体育教学辅助系统，提供课堂行为监测、运动安全告警、训练动作分析等功能。

## 项目简介

本项目以**智慧体育教学服务**为核心，视觉算法为辅助技术，旨在解决体育课堂中的安全防护与教学质量评估问题。系统采用「YOLOv8 多人检测主引擎 + BlazePose 单人精细分析辅助」的双层架构，实现课堂全员实时监测与风险人员深度确认的有机结合。

### 核心能力

- **多人实时检测**：YOLOv8-pose 实现课堂全员姿态估计，COCO 17关键点
- **单人精细分析**：BlazePose 33关键点 + 3D信息，对告警人员自动启动深度分析
- **风险动作识别**：弯腰过猛、膝盖过脚尖、屈膝过度、膝盖内扣、深蹲塌腰、手臂过伸、颈部后仰等
- **课堂行为识别**：举手、蹲下、懒散站姿、叉腰等课堂状态
- **四场景差异化**：校园体育、健身训练、康复训练、通用模式，各场景检测项与阈值不同

### 场景差异化

| 场景 | 检测重点 | 风险灵敏度 | 特色功能 |
|:---|:---|:---|:---|
| 校园体育 | 课堂行为、参与度、安全告警 | 标准 | 到课统计、行为识别 |
| 健身训练 | 动作识别、姿势纠正 | 严格 | 深蹲/开合跳/高抬腿识别 |
| 康复训练 | 关节活动度、温和提示 | 宽松 | 活动范围监测 |
| 通用模式 | 全部检测项 | 标准 | 多场景适配 |

## 系统架构

```
┌─────────────────────────────────────────────────────┐
│                    前端（H5 响应式）                   │
│  手机/电脑浏览器 · 实时摄像头WebSocket · 视频上传分析   │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP/WebSocket
┌──────────────────────┴──────────────────────────────┐
│                  后端（FastAPI）                      │
│  用户认证 · 视频处理 · WebSocket实时分析 · 数据库管理   │
├──────────────┬───────────────┬──────────────────────┤
│  YOLOv8-pose │   BlazePose   │     ffmpeg转码        │
│  多人检测主引擎│  单人精细分析  │  H.264浏览器兼容      │
└──────────────┴───────────────┴──────────────────────┘
```

## 文件结构

```
.
├── web_server.py              # Web服务端（FastAPI + WebSocket）
├── sports_classroom_system.py # 核心检测引擎（YOLOv8 + BlazePose整合）
├── yolov8_classroom_monitor.py# YOLOv8多人检测模块
├── blazepose_sports_demo.py   # BlazePose单人分析模块
├── client_app.py              # PyQt5桌面客户端
├── static/
│   └── index.html             # 前端H5页面
├── 校园智慧体育课堂项目计划书.docx  # 项目计划书
└── build_exe.py               # 打包脚本
```

## 快速开始

### 环境要求

- Python 3.8+
- 摄像头（实时监测功能）
- ffmpeg（视频转码，需在PATH中）

### 安装依赖

```bash
pip install ultralytics mediapipe opencv-python numpy fastapi uvicorn python-multipart pydantic cryptography
```

### 下载模型权重

```bash
# YOLOv8-pose 权重会由ultralytics自动下载
# BlazePose 模型需手动下载
mkdir models
# 下载 pose_landmarker_lite.task 到 models/ 目录
# 下载地址: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

### 启动Web服务

```bash
# 基本启动（本机访问）
python web_server.py

# 启用HTTPS（手机IP访问时摄像头需要）
python web_server.py --https

# 启用公网穿透（异地访问）
python web_server.py --https --tunnel cloudflare

# 指定端口
python web_server.py --port 8080
```

启动后访问 `http://127.0.0.1:8000`（HTTP）或 `https://127.0.0.1:8000`（HTTPS）。

### 启动桌面客户端

```bash
pip install PyQt5
python client_app.py
```

## 技术栈

| 技术 | 用途 |
|:---|:---|
| YOLOv8-pose | 多人实时姿态检测（COCO 17点） |
| MediaPipe BlazePose | 单人精细动作分析（33点 + 3D） |
| FastAPI | Web后端服务 + WebSocket实时通信 |
| OpenCV | 视频处理 + 图像渲染 |
| ffmpeg | 视频转码（H.264浏览器兼容） |
| SQLite | 用户数据与监测记录存储 |
| Cloudflare Tunnel | 免注册公网穿透 |

## 安全设计

- 用户密码 PBKDF2 + 随机盐值哈希存储
- Token-based 认证，所有API请求需携带用户令牌
- 用户数据隔离：视频文件按 `uploads/{用户ID}/` 和 `results/{用户ID}/` 存储
- 数据库按用户ID过滤，各用户只能查看自己的记录

## 项目定位

**主体**：智慧体育教学服务（课堂监测、安全防护、教学减负）
**辅助**：YOLOv8 负责多人实时感知，BlazePose 负责单人风险精细确认

> 技术为教学服务，而非为技术而技术。系统设计的出发点是解决体育课堂中的实际问题，视觉算法是实现手段而非目的。

## License

MIT
