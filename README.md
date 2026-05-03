# 🏢 Intern @ Connected Tech — AI Engineer (Jul–Sep 2025)

<div align="center">

<img src="https://raw.githubusercontent.com/VarakronVi/intern-connected-tech/main/assets/connected-tech-logo.png" width="300"/>

<br><br>

![Internship](https://img.shields.io/badge/AI%20Engineer%20Intern-Connected%20Tech-1a1a2e?style=for-the-badge)
![Duration](https://img.shields.io/badge/Duration-2%20Months-orange?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLOv11](https://img.shields.io/badge/YOLOv11-Object%20Detection-FF6B35?style=for-the-badge&logo=opencv&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)

**2-month AI Engineering internship building a real-time factory safety system**

*Showcased at **Super AI Engineer Season 5** — The 5th National AI Exhibition, Central Rama 9*

---

</div>

## 📋 Project — IndusGuard: Factory Vehicle Speed Monitoring System

### 🎬 System in Action

<div align="center">
<img src="https://raw.githubusercontent.com/VarakronVi/intern-connected-tech/main/assets/demo.jpg" width="700"/>

*Real-time speed detection — bounding box, speed display (km/h), status (NORMAL/VIOLATION), and ROI polygon*
</div>

## 📌 Overview

**IndusGuard** is a production-ready computer vision system built to monitor and regulate vehicle speeds within factory premises using existing CCTV infrastructure. The system detects vehicles in real time, estimates their speed through homography-based perspective transformation, and automatically records evidence when a speed violation occurs.

> **Problem**: Over **40%** of workplace accidents in Thailand involve vehicles. Factory environments — forklifts, motorcycles, trucks — lack speed enforcement systems, creating serious safety risks.

> **Solution**: IndusGuard transforms standard CCTV into a smart safety layer — no new hardware, no model weights distributed, just intelligent software on top of existing infrastructure.

---

## 🎯 Key Features

| Feature | Description |
|---|---|
| 🎥 **Real-time Detection** | Detects cars, motorcycles, and forklifts (Hooklift) via YOLOv11 |
| 📐 **Perspective Calibration** | Homography matrix maps pixel distances to real-world meters |
| 🏃 **Multi-object Tracking** | Hungarian Algorithm with deque-based velocity smoothing |
| 🚨 **Speed Enforcement** | Compares against configurable zone speed limits |
| 📼 **Evidence Recording** | Automatically clips and saves violation footage |
| ⚡ **Async Architecture** | Non-blocking pipeline using `asyncio` + `threading` + `Queue` |
| 🔧 **YAML Configuration** | All parameters (zones, thresholds, paths) managed via config files |

---

## 🏗️ System Architecture

```
Input (CCTV Stream)
        │
        ▼
┌─────────────────┐
│  YOLO Detection │  ← YOLOv11n — Cars, Motorcycles, Hooklift
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│ Homography Calibrate│  ← Perspective Transform: pixel → meters
└────────┬────────────┘
         │
         ▼
┌────────────────────┐
│  Hungarian Tracker │  ← Frame-to-frame ID assignment
└────────┬───────────┘
         │
         ▼
┌──────────────────────┐
│  Speed Calculation   │  ← distance / time + Outlier Filter
└────────┬─────────────┘
         │
         ▼
┌───────────────────────┐
│  Enforcement Engine   │  ← Check against speed limit per zone
└────────┬──────────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
 Alert    Evidence
(notify)  (video clip)
```

---

## 🛠️ Tech Stack

- **Detection**: `ultralytics` YOLOv11 — custom-trained on factory vehicle dataset
- **Calibration**: `scipy.optimize` + `numpy` for homography computation
- **Tracking**: Hungarian Algorithm via `scipy.optimize.linear_sum_assignment`
- **Speed Estimation**: Euclidean distance over time using `collections.deque` smoothing
- **Async Engine**: `asyncio`, `threading`, `queue.Queue`
- **Config Management**: `yaml` + `pathlib`
- **Logging**: Structured logging with `logging` + `datetime` timestamps
- **Evidence**: `cv2.VideoWriter` — violation-only clip recording

---

## 📁 Project Structure

```
speed-estimation/
├── src/
│   ├── app/
│   ├── assets/
│   │   ├── images/
│   │   └── videos/              # Test footage (notover20km, Speed_motor, etc.)
│   ├── backend/
│   ├── frontend/
│   └── ui/
│       ├── app.py
│       ├── get_polygon.py       # ROI calibration utility
│       └── noolee_explore.py
├── test/
│   ├── car_speed_tracker_with_config.py
│   ├── motocycle.py
│   └── speed_estimation.py     # Main system (EvidenceRecorder + VehicleSpeedTracker)
├── utils/
│   ├── data_utils/
│   ├── io_utils/
│   ├── logging_utils/
│   └── string_utils/
└── README.md
```

---

## ⚙️ Configuration (YAML-Driven)

All system parameters are externalized — no hardcoded values:

```yaml
# Example config structure
detection:
  model_path: "assets/models/yolo11n.pt"
  confidence: 0.5
  classes: [car, motorcycle, hooklift]

zones:
  - id: zone_a
    speed_limit_kmh: 20
    polygon: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

homography:
  reference_points: 4
  real_world_distance_m: 10.0

evidence:
  output_dir: "evidence/"
  pre_buffer_frames: 30
  post_buffer_frames: 60
```

---

## 🧪 Object Detection Model

| Model | Classes Tested | mAP@0.5:0.95 | mAP@0.5 | Precision | Recall |
|-------|---------------|--------------|---------|-----------|--------|
| YOLOv11n | Hooklift (custom) | 0.985 | 0.995 | 0.9999 | 0.9991 |

> **Training**: 5,000 augmented images (10 raw → background removal → 2,500 → augmentation → 5,000), 150 epochs, 640×640

> **Note**: Model weights are not distributed in this repository. Only the detection + tracking pipeline is included.

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.10+
pip install ultralytics opencv-python scipy numpy pyyaml
```

### Run

```bash
# Calibrate ROI polygon on your CCTV feed
python src/ui/get_polygon.py

# Run speed tracker (configure your YAML first)
python test/speed_estimation.py --config config/factory_zone.yaml
```

---

## 📊 Results

- ✅ Real-time processing on standard CPU (no GPU required for inference)
- ✅ Violation evidence recorded only when speed limit exceeded (storage-efficient)
- ✅ Multiple vehicle classes tracked simultaneously in the same frame
- ✅ Outlier filtering prevents false alerts from tracking jitter

---

## 👤 Author

**Varakron Vimolgarnjana** — AI Engineer Intern @ Connected Tech

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/varakron-vimolgarnjana-b4b36b2bb/)
[![GitHub](https://img.shields.io/badge/GitHub-@VarakronVi-181717?style=flat&logo=github)](https://github.com/VarakronVi)

---

## 🏆 Recognition

This project was exhibited at the **Super AI Engineer Season 5 — The 5th National AI Exhibition** at Central Rama 9, Bangkok, demonstrating real-world AI solutions for industrial safety under the **Connected Tech** team.

---

<div align="center">
<sub>Built with 🔧 during a 2-month AI Engineering internship | Industrial Safety · Computer Vision · Real-time Systems</sub>
</div>
