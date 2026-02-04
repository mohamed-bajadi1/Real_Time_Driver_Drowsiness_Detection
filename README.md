# Real-Time Driver Drowsiness Detection System 🚗💤

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8.1-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.9-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**An intelligent real-time drowsiness detection system using computer vision and deep learning to prevent accidents caused by driver fatigue.**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [Performance](#-performance) • [Documentation](#-documentation)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Technical Details](#-technical-details)
- [Performance Metrics](#-performance-metrics)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

The Real-Time Driver Drowsiness Detection System is an advanced computer vision solution designed to monitor driver alertness and prevent accidents caused by drowsiness. The system analyzes facial features in real-time using a combination of MediaPipe facial landmark detection and a custom CNN-based eye state classifier.

### Key Capabilities:
- **Real-time eye state detection** - Distinguishes between open and closed eyes with 99% accuracy
- **Yawn detection** - Uses Mouth Aspect Ratio (MAR) algorithm for fatigue detection
- **Head pose estimation** - Detects head drooping and abnormal head positions
- **Multi-signal fusion** - Combines CNN predictions with geometric features (EAR - Eye Aspect Ratio)
- **Smart alarm system** - Temporal smoothing to prevent false alarms from natural blinking
- **Performance optimized** - Achieves 20-30 FPS on standard hardware

---

## ✨ Features

### 🔍 Detection Capabilities

| Feature | Description | Status |
|---------|-------------|--------|
| **Eye State Classification** | CNN-based binary classification (open/closed) | ✅ Production |
| **Yawn Detection** | MAR-based fatigue detection | ✅ Production |
| **Head Pose Estimation** | 3D head orientation tracking (pitch, yaw, roll) | ✅ Production |
| **Geometric Head Droop** | Detects sideways and forward head drooping | ✅ Production |
| **Temporal Smoothing** | Prevents false alarms from natural blinks | ✅ Production |
| **Multi-Modal Fusion** | Combines CNN + EAR for robust predictions | ✅ Production |

### 🚨 Alarm Triggers

The system triggers visual and audio alerts when:
- Both eyes closed for **20 consecutive frames** (~0.7-1 second)
- Eyes closed + recent yawn detected → **Faster alarm** (10 frames)
- Excessive yawning (≥5 yawns + ≥6 yawns/minute)
- Head drooping (pitch < -30° for 60 frames)
- Severe head droop (pitch < -45° for 30 frames) → **Immediate alarm**

### 🎨 Visualization Features

- Real-time FPS counter
- Eye state indicators (OPEN/CLOSED with confidence scores)
- Yawn detection status and frequency tracking
- Head pose angles visualization
- Performance metrics dashboard
- Debug mode with preprocessed eye images

---

## 🏗️ System Architecture

### Detection Pipeline

```
┌─────────────────┐
│  Camera Input   │
│   640×480 @ 30  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  MediaPipe      │
│  FaceMesh       │ ◄── 478 facial landmarks
└────────┬────────┘
         │
         ├──────────────┬──────────────┬──────────────┐
         ▼              ▼              ▼              ▼
    ┌────────┐    ┌─────────┐   ┌─────────┐   ┌──────────┐
    │ Eye    │    │ Yawn    │   │ Head    │   │ Temporal │
    │ CNN    │    │ MAR     │   │ Pose    │   │ Filter   │
    └────┬───┘    └────┬────┘   └────┬────┘   └────┬─────┘
         │             │             │             │
         └─────────────┴─────────────┴─────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Fusion Engine   │
              │ Decision Logic  │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Alarm System    │
              │ Visual + Audio  │
              └─────────────────┘
```

### CNN Architecture

The eye state classifier uses a compact CNN architecture:

```python
Model: "eye_state_classifier"
_________________________________________________________________
Layer (type)                 Output Shape              Params   
=================================================================
conv2d (Conv2D)              (None, 62, 62, 16)        160      
max_pooling2d (MaxPooling2D) (None, 31, 31, 16)        0        
batch_normalization          (None, 31, 31, 16)        64       
dropout (Dropout)            (None, 31, 31, 16)        0        
                                                                 
conv2d_1 (Conv2D)            (None, 29, 29, 32)        4,640    
max_pooling2d_1              (None, 14, 14, 32)        0        
batch_normalization_1        (None, 14, 14, 32)        128      
dropout_1 (Dropout)          (None, 14, 14, 32)        0        
                                                                 
conv2d_2 (Conv2D)            (None, 12, 12, 64)        18,496   
max_pooling2d_2              (None, 6, 6, 64)          0        
batch_normalization_2        (None, 6, 6, 64)          256      
dropout_2 (Dropout)          (None, 6, 6, 64)          0        
                                                                 
flatten (Flatten)            (None, 2304)              0        
dense (Dense)                (None, 64)                147,520  
dropout_3 (Dropout)          (None, 64)                0        
dense_1 (Dense)              (None, 1)                 65       
=================================================================
Total params: 171,329
Trainable params: 171,105
Non-trainable params: 224
```

**Input:** 64×64 grayscale images  
**Output:** Binary classification (0 = closed, 1 = open)  
**Preprocessing:** CLAHE contrast enhancement + normalization

---

## 🔧 Installation

### Prerequisites

- Python 3.10
- Webcam or camera device
- 4GB+ RAM recommended

### Step 1: Clone the Repository

```bash
git clone https://github.com/mohamed-bajadi1/Real_Time_Driver_Drowsiness_Detection.git
cd Real_Time_Driver_Drowsiness_Detection
```

### Step 2: Create Virtual Environment

```bash
# Windows
py -3.10 -m venv venv
venv\Scripts\activate

# macOS/Linux
python3.10 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Setup

```bash
cd src
python verify_setup.py
```

You should see:
```
✅ OpenCV - OK
✅ MediaPipe - OK
✅ TensorFlow - OK
✅ NumPy - OK
✅ Camera (index 0) - OK
```

---

## 🚀 Usage

### Quick Start

Run the latest version with all features:

```bash
cd src
python main_v5_3_geometric.py
```

### Version Comparison

| Script | Features | FPS | Use Case |
|--------|----------|-----|----------|
| `main.py` | Basic landmark detection | 25-30 | Testing MediaPipe setup |
| `main_v2.py` | + CNN eye classification | 8-12 | Early prototype |
| `main_v3.py` | + CLAHE + temporal smoothing | 10-15 | Improved stability |
| `main_v4.py` | + Batched inference | 15-20 | Performance optimization |
| `main_v5_2_head_pose.py` | + PnP head pose | 12-18 | 3D head tracking |
| **`main_v5_3_geometric.py`** | + Geometric head droop | **20-30** | **✅ Recommended** |

### Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit application |
| `d` | Toggle debug mode (show preprocessed eye images) |
| `t` | Adjust detection threshold |
| `s` | Toggle skip-frame mode |
| `h` | Toggle head pose visualization |

### Configuration

Edit the configuration section in the script:

```python
# Camera settings
CAMERA_INDEX = 0              # Change if using external camera
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Detection thresholds
DETECTION_THRESHOLD = 0.4     # Lower = more sensitive (0.3-0.5)
DROWSINESS_FRAME_THRESHOLD = 20  # Frames before alarm

# Audio settings
ENABLE_AUDIO = True
ALARM_VOLUME = 0.7

# Performance
USE_TFLITE = True             # Use optimized TFLite model
SKIP_FRAME_MODE = False       # Process every Nth frame
```

---

## 📁 Project Structure

```
Real_Time_Driver_Drowsiness_Detection/
│
├── src/                              # Source code
│   ├── main.py                       # v1 - Basic FaceMesh
│   ├── main_v2.py                    # v2 - CNN integration
│   ├── main_v3.py                    # v3 - CLAHE + smoothing
│   ├── main_v4.py                    # v4 - Batched inference
│   ├── main_v5_2_head_pose.py        # v5.2 - PnP head pose
│   ├── main_v5_3_geometric.py        # v5.3 - Geometric droop detection ⭐
│   │
│   ├── enhanced_eye_detector.py      # Multi-modal eye state detector
│   ├── head_pose_estimator.py        # PnP-based head pose estimation
│   ├── geometric_head_droop_detector.py  # Geometric droop detection
│   ├── landmark_reference.py         # MediaPipe landmark documentation
│   │
│   ├── preprocess_data.py            # Dataset preprocessing
│   ├── verify_setup.py               # Environment verification
│   └── demo_extract_eye.py           # Eye extraction demo
│
├── models/                           # Trained models
│   ├── eye_state_classifier_v3.h5    # CNN model (Keras)
│   ├── eye_state_classifier_v4.h5    # v4 robust model
│   └── eye_state_classifier_v4.tflite # TFLite optimized
│
├── data/                             # Data and assets
│   ├── eyes/                         # Preprocessed eye images
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── sounds/                       # Alarm sounds
│       └── alarm.wav
│
├── notebooks/                        # Jupyter notebooks
│   └── train-eye-model.ipynb         # Model training notebook
│
├── docs/                             # Documentation
��   ├── DROWSINESS_SYSTEM_ANALYSIS.md # System architecture analysis
│   └── README_Eye_Extraction.md      # Eye extraction guide
│
├── results/                          # Output visualizations
│
├── requirements.txt                  # Python dependencies
├── .gitignore
├── LICENSE
└── README.md                         # This file
```

---

## 🔬 Technical Details

### Eye State Detection

**Method:** Hybrid approach combining CNN appearance-based classification with EAR geometric features

#### CNN Prediction
- **Input:** 64×64 grayscale eye image with CLAHE preprocessing
- **Architecture:** 3-block CNN with batch normalization and dropout
- **Output:** P(open) ∈ [0, 1]
- **Threshold:** < 0.4 = CLOSED, ≥ 0.4 = OPEN

#### Eye Aspect Ratio (EAR)
```python
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
```
Where p1...p6 are eye landmark coordinates.

- **Threshold:** EAR < 0.21 = CLOSED

#### Fusion Strategy
```python
if CNN_confidence > 80%:
    final_state = CNN_prediction
elif abs(CNN - EAR) < threshold:
    final_state = (CNN + EAR) / 2  # Agreement
else:
    final_state = CNN_prediction    # CNN more reliable
```

### Yawn Detection

**Method:** Mouth Aspect Ratio (MAR)

```python
MAR = vertical_distance / horizontal_distance
```

- Vertical: Distance between upper and lower lip centers
- Horizontal: Distance between left and right mouth corners
- **Threshold:** MAR > 0.6 for 3+ consecutive frames = YAWN

### Head Droop Detection

**v5.3 Geometric Method** (Current):
- Measures eye line tilt angle
- Tracks face vertical position
- Detects sideways drooping (PnP cannot)
- **Performance:** < 1ms per frame

**v5.2 PnP Method** (Legacy):
- 6-point Perspective-n-Point algorithm
- Extracts Euler angles (pitch, yaw, roll)
- **Limitation:** Fails for sideways head droop
- **Performance:** ~3ms per frame

### Preprocessing Pipeline

```python
1. Extract eye region from landmarks → bbox
2. Add padding (15px) for context
3. Resize to 64×64
4. Convert to grayscale
5. Apply CLAHE (clip=2.0, tile=4×4)
6. Normalize to [0, 1]
7. Add batch dimension → (1, 64, 64, 1)
```

---

## 📊 Performance Metrics

### Model Performance

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| **Accuracy** | 99.2% | 98.8% | 99.1% |
| **Precision** | 99.3% | 98.9% | 99.2% |
| **Recall** | 99.1% | 98.7% | 99.0% |
| **F1-Score** | 99.2% | 98.8% | 99.1% |

### Inference Performance

| Configuration | FPS | Frame Time | Hardware |
|---------------|-----|------------|----------|
| v3 (Sequential) | 5-10 | 100-200ms | CPU |
| v4 (Batched) | 12-15 | 65-85ms | CPU |
| v4 + TFLite | 20-30 | 33-50ms | CPU |
| v5.3 (Geometric) | 20-30 | 33-50ms | CPU |

**Test Environment:**
- Intel Core i5 (8th gen) / AMD Ryzen 5
- 8GB RAM
- No GPU acceleration
- 640×480 resolution

### Performance Breakdown

| Component | Time (ms) | % of Frame |
|-----------|-----------|------------|
| Camera read | 2-3 | 6% |
| MediaPipe FaceMesh | 15-25 | 45% |
| CNN inference (batched) | 8-12 | 25% |
| Yawn detection (MAR) | <1 | 2% |
| Head droop (geometric) | <1 | 2% |
| Rendering | 5-8 | 15% |
| **Total** | **33-50** | **100%** |

---

## 🗺️ Roadmap

### Completed ✅

- [x] MediaPipe FaceMesh integration
- [x] CNN eye state classifier (99% accuracy)
- [x] CLAHE preprocessing pipeline
- [x] Temporal smoothing and false alarm reduction
- [x] Batched inference optimization
- [x] TensorFlow Lite conversion
- [x] Yawn detection (MAR-based)
- [x] Head pose estimation (PnP)
- [x] Geometric head droop detection
- [x] Multi-modal fusion (CNN + EAR)
- [x] Smart alarm system with cooldown

### In Progress 🚧

- [ ] Mobile deployment (Android/iOS)
- [ ] Edge device optimization (Raspberry Pi, Jetson Nano)
- [ ] Advanced temporal models (LSTM for drowsiness progression)

### Planned 📋

- [ ] Multi-person detection in vehicles
- [ ] Driver identification
- [ ] Distraction detection (phone usage, not looking at road)
- [ ] Cloud logging and analytics dashboard
- [ ] Integration with vehicle CAN bus
- [ ] Explainable AI visualization

---

## 📚 Documentation

### Additional Resources

- **[DROWSINESS_SYSTEM_ANALYSIS.md](./docs/DROWSINESS_SYSTEM_ANALYSIS.md)** - Detailed architectural analysis and optimization roadmap
- **[README_Eye_Extraction.md](./docs/README_Eye_Extraction.md)** - Eye extraction methodology and best practices
- **[Training Notebook](./notebooks/train-eye-model.ipynb)** - Model training walkthrough

### Key Findings

From the system analysis:
- **Bottleneck:** Sequential CNN inference accounted for 64% of frame time
- **Solution:** Batched inference reduced CNN time from 112ms to 20ms (5.6× speedup)
- **Critical Bug:** Training without CLAHE caused distribution mismatch → poor real-world performance
- **Fix:** Applied CLAHE consistently in both training and inference pipelines

### MediaPipe Landmarks Used

| Feature | Landmark Indices | Count |
|---------|-----------------|-------|
| Left Eye | 33, 133, 160, 159, 158, 144, 145, 153 | 8 |
| Right Eye | 362, 263, 387, 386, 385, 373, 374, 380 | 8 |
| Mouth | 61, 291, 0, 17, 269, 405, 314, etc. | 12 |
| Head Pose | 1, 152, 33, 263, 61, 291 | 6 |

See [`landmark_reference.py`](./src/landmark_reference.py) for complete documentation.

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Reporting Bugs

1. Check existing [Issues](https://github.com/mohamed-bajadi1/Real_Time_Driver_Drowsiness_Detection/issues)
2. Create a new issue with:
   - System information (OS, Python version, hardware)
   - Steps to reproduce
   - Expected vs actual behavior
   - Screenshots/logs if applicable

### Suggesting Features

Open an issue with the `enhancement` label and describe:
- The problem your feature solves
- Proposed implementation approach
- Any related work or references

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Add tests if applicable
5. Commit with clear messages (`git commit -m 'Add AmazingFeature'`)
6. Push to your fork (`git push origin feature/AmazingFeature`)
7. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add docstrings to functions and classes
- Update documentation for new features
- Ensure backward compatibility when possible

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Mohamed Bajadi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 🙏 Acknowledgments

### Datasets

- **Closed Eyes in the Wild (CEW)** - Eye state classification dataset
- **MRL Eye Dataset** - Additional training data for robustness

### Libraries and Frameworks

- [MediaPipe](https://google.github.io/mediapipe/) - Google's ML solutions for face detection
- [TensorFlow](https://www.tensorflow.org/) - Deep learning framework
- [OpenCV](https://opencv.org/) - Computer vision library
- [NumPy](https://numpy.org/) - Scientific computing

### Inspiration

This project was inspired by the need to prevent drowsy driving accidents, which account for approximately 20% of all traffic accidents worldwide according to the National Highway Traffic Safety Administration (NHTSA).

### Team

- **Binomial Team** - Development and research
- **Claude (Anthropic)** - System architecture analysis and optimization recommendations

---

## 📞 Contact

**Mohamed Bajadi** & **Mustapha Zmirli**

- GitHub: [@mohamed-bajadi1](https://github.com/mohamed-bajadi1) - [@Mustapha-Zmirli](https://github.com/Mustapha-Zmirli)
- Project Link: [https://github.com/mohamed-bajadi1/Real_Time_Driver_Drowsiness_Detection](https://github.com/mohamed-bajadi1/Real_Time_Driver_Drowsiness_Detection)

---

## 📈 Statistics

![GitHub Stars](https://img.shields.io/github/stars/mohamed-bajadi1/Real_Time_Driver_Drowsiness_Detection?style=social)
![GitHub Forks](https://img.shields.io/github/forks/mohamed-bajadi1/Real_Time_Driver_Drowsiness_Detection?style=social)
![GitHub Issues](https://img.shields.io/github/issues/mohamed-bajadi1/Real_Time_Driver_Drowsiness_Detection)
![GitHub Pull Requests](https://img.shields.io/github/issues-pr/mohamed-bajadi1/Real_Time_Driver_Drowsiness_Detection)

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

**Made with ❤️ for road safety**

</div>
