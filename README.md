# The Aerial Guardian: Drone Detection & Tracking Pipeline

A lightweight, custom-engineered **Multi-Object Tracking (MOT)** pipeline designed specifically for aerial drone surveillance platforms.  
This system is optimized to handle the unique challenges of drone-based vision systems, including:

- Tiny object footprints at high altitude
- Persistent camera ego-motion
- Motion blur and temporary occlusions
- Limited onboard compute resources

The pipeline combines **YOLOv8-based detection**, **Global Motion Compensation (GMC)**, and **ByteTrack-inspired tracking logic** to deliver stable and efficient object tracking in aerial environments.

---

# Project Highlights

✅ Real-time aerial object detection and tracking  
✅ Custom Global Motion Compensation (GMC) module  
✅ ByteTrack-inspired multi-stage association logic  
✅ Kalman Filter-based trajectory prediction  
✅ Lightweight architecture (~22.5 MB model size)  
✅ Optimized for resource-constrained drone systems  

---

# System Architecture

The pipeline consists of three tightly integrated modules:

```text
Input Drone Video
        │
        ▼
YOLOv8s Object Detection
        │
        ▼
Global Motion Compensation (GMC)
        │
        ▼
ByteTrack-style Data Association
        │
        ▼
Kalman Filter Prediction
        │
        ▼
Final Tracked Output Video
```

---

# Core Engineering Contributions

## Small Object Detection (YOLOv8s Baseline)

### Problem
In aerial surveillance, targets appear extremely small due to altitude.  
For example, a pedestrian may occupy fewer than:

:contentReference[oaicite:0]{index=0}

pixels in a frame.

Traditional detectors often lose fine spatial details during downsampling.

---

### Solution

We use the anchor-free **YOLOv8s (Small)** model as the primary detector.

### Key Optimizations

- Lightweight architecture (~22.5 MB)
- High-resolution inference canvas:
  
:contentReference[oaicite:1]{index=1}

- Better preservation of spatial information
- Improved recall for tiny moving targets

### Why YOLOv8s?

| Feature | Benefit |
|---|---|
| Anchor-Free Design | Better localization flexibility |
| Lightweight Model | Faster deployment on edge hardware |
| Strong Small-Object Recall | Suitable for aerial surveillance |
| Easy Integration | Works seamlessly with tracking pipeline |

---

# Camera Noise & Ego-Motion Compensation (GMC)

## The Challenge

Drone movement introduces massive background displacement between consecutive frames.

When the drone:
- pitches
- rolls
- yaws

the entire scene shifts globally.

Traditional trackers incorrectly interpret this background motion as actual object movement, leading to:
- ID switching
- track fragmentation
- unstable trajectories

---

##  Our Custom GMC Engine

Implemented in:

```bash
src/utils/gmc.py
```

The module performs:

### Step 1 — ORB Feature Extraction
Background keypoints are extracted using:
- ORB detector
- descriptor matching

### Step 2 — RANSAC Outlier Rejection
Foreground moving objects are rejected using:
- affine consistency filtering
- RANSAC-based estimation

### Step 3 — Affine Motion Estimation

The system computes an affine transformation matrix:

:contentReference[oaicite:2]{index=2}

which models global camera movement between frames.

### Step 4 — Motion Compensation

The inverse transform is applied to tracker states so that:
- stationary objects remain stable
- camera shake is neutralized
- tracking consistency improves dramatically

---

# High-Recall Multi-Object Tracking (ByteTrack Logic)

## The Problem

Aerial videos frequently suffer from:
- motion blur
- occlusions
- low-confidence detections
- intermittent target disappearance

Most trackers immediately discard weak detections, causing:
- identity loss
- unstable tracks

---

## Our Tracking Strategy

The system uses a ByteTrack-inspired dual-threshold association mechanism.

### Core Components

### IoU-Based Matching

Bounding boxes are matched using:

:contentReference[oaicite:3]{index=3}

This helps associate detections across frames.

---

### Kalman Filter Prediction

When detections temporarily disappear, the tracker predicts object positions using a linear Kalman Filter.

State propagation follows:

:contentReference[oaicite:4]{index=4}

This enables:
- smooth trajectory continuation
- temporary occlusion handling
- stable identity preservation

---

### Track Retention Logic

Tracks are retained for:

```python
max_age = 30
```

frames before deletion.

This allows the system to recover identities when targets reappear after short-term occlusion.

---

# Project Structure

```bash
AERIAL-GUARDIAN/
│
├── data/
│   ├── annotations/
│   ├── sequences/
│   ├── test_input.mp4
│   └── output_tracked.mp4
│
├── src/
│   └── utils/
│       └── gmc.py
│
├── weights/
│
├── create_input_video.py
├── inference.py
├── train.py
├── yolov8s.pt
├── README.md
└── .gitignore
```

---

# Performance & Telemetry

| Metric | Value |
|---|---|
| Model Size | ~22.5 MB |
| Inference Resolution | 1280px |
| Tracking Type | Multi-Object Tracking |
| Tracking Logic | ByteTrack-inspired |
| Motion Compensation | Custom GMC |
| Hardware Tested | Intel Core i7 / 16GB RAM |
| Execution Throughput | ~1 FPS (CPU) |

---

# Output Video

Google Drive Output Samples:

🔗 https://drive.google.com/drive/folders/1xK9Li9nAOsJJkuz_n7iWnpkdjLaFF-mF

The output videos demonstrate:
- stable ID tracking
- motion compensation
- reduced identity switching
- smoother aerial object trajectories

---

# Installation

Clone the repository and install dependencies:

```bash
git clone <your-repo-link>

cd AERIAL-GUARDIAN
```

Install required packages:

```bash
pip install ultralytics filterpy opencv-python numpy
```

---

# Running Inference

Run the tracking pipeline using:

```bash
python inference.py
```

The processed output video will be saved as:

```bash
output_tracked.mp4
```

---

# Training

To train the YOLOv8 detector on a custom dataset:

```bash
python train.py
```

You can modify:
- dataset path
- epochs
- batch size
- image resolution

inside the training script.

---

# Research & Engineering Focus

This project was designed with emphasis on:

- aerial computer vision
- embedded AI systems
- robust multi-object tracking
- low-resource deployment
- real-world drone surveillance scenarios

The pipeline demonstrates how lightweight detection systems can be significantly improved through carefully engineered motion compensation and association strategies.

---

# Author

**Ankita Raj**

B.Tech CSE | Full Stack & AI Developer  
Focused on:
- Computer Vision
- AI Systems
- Full Stack Development
- Intelligent Surveillance Systems

---

# License

This project is intended for educational, research, and portfolio purposes.
