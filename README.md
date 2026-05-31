# 🛸 The Aerial Guardian: Drone Detection & Tracking Pipeline

A lightweight, custom-engineered Multi-Object Tracking (MOT) pipeline designed specifically for aerial drone surveillance platforms. This architecture addresses the unique challenges of high-altitude computer vision: small target footprints, persistent background motion ("camera noise"), and strict hardware resource ceilings.

---

## 🚀 System Architecture & Key Contributions

Unlike generic out-of-the-box object trackers that fail under drone ego-motion, this solution custom-orchestrates three core engineering components:

### 1. Small Object Detection (YOLOv8s Baseline)
* **The Architecture:** We utilize the anchor-free **YOLOv8s (Small)** model as our base feature extractor. It maintains an exceptionally light footprint of **~22.5 MB**, which is comfortably below the project's strict 300 MB limit.
* **The Optimization:** To counter target degradation at high altitudes (where a pedestrian may span less than a $15 \times 15$ pixel area), the pipeline dynamically processes frames on an upscaled **1280px inference canvas (`imgsz=1280`)**. This configuration forces the early convolutional layers to retain high spatial resolution for dense pixel clusters before downsampling occurs.

### 2. Camera Noise & Ego-Motion Compensation (GMC)
* When a drone pitches, yaws, or rolls, the static background environment shifts drastically. Standard tracking systems interpret this global pixel translation as true target velocity, breaking existing tracks and generating widespread identity switching.
* **Our Core Addition:** We implemented a custom **Global Motion Compensation (GMC)** engine (`src/utils/gmc.py`). The module extracts background keypoints frame-by-frame using an **ORB feature detector**, screens out moving foreground entities via **RANSAC outlier rejection**, and computes a robust **Affine Transformation Matrix**. This matrix is mathematically inverted and fed into the tracking loops to step-compensate the tracker's internal spatial coordinate memory—effectively "nullifying" drone flight movement.

### 3. High-Recall Multi-Object Tracking (ByteTrack Logic)
* Drone video streams frequently introduce motion blur or momentary object occlusions (e.g., pedestrians walking beneath tree canopies or traffic signs). 
* The system utilizes a dual-threshold data association layout inspired by **ByteTrack**. Instead of immediately discarding low-confidence bounding boxes, it maps them across consecutive frames using an **Intersection-over-Union (IoU)** matching matrix linked to a linear **Kalman Filter**. If a target vanishes, the filter projects its movement vector forward for up to 30 frames (`max_age=30`), gracefully retaining the **exact same ID** once the subject re-emerges.

---

## ⚙️ Performance & Telemetry Results

* **Model File Size:** ~22.5 MB (`yolov8s.pt`) — *Well within the 300 MB constraint.*
* **Test Hardware Setup:** Host CPU Inference (Intel Core i7 / 16GB RAM)
* **Execution Throughput:** ~1.0 FPS *(Baseline CPU evaluation rate processing a dense 1280px canvas)*

---

## 🛠️ Setup & Execution Guide

### 1. Installation
Clone this repository and install the standard dependencies:
```bash
pip install ultralytics filterpy opencv-python numpy
