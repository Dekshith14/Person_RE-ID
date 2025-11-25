
# Person Re-Identification System

### Gait + Appearance + Body Ratio Fusion • YOLOv8 Tracking • Streamlit Frontend

This project implements a complete Person Re-Identification (Re-ID) pipeline that combines gait features, appearance features, and biometric body ratios. The system performs multi-person tracking and identification from video input and provides a simple web interface using Streamlit.

---

## Features

* YOLOv8-Seg for person detection and tracking
* Gait recognition using a CNN-LSTM model
* Appearance embedding extraction using OSNet (Torchreid)
* Body ratio features from YOLOv8-Pose
* Three-way feature fusion (770 dimensions total)
* Identification against enrolled gallery embeddings
* Streamlit-based web app for uploading videos and viewing results

---

## Project Structure

```
person_reid_app/
├── app.py                  # Streamlit frontend
├── reid_pipeline.py        # Inference pipeline
├── requirements.txt
├── models/
│   └── my_gait_cnnlstm.pth
├── gallery/
│   └── my_known_gallery.pth
└── outputs/                # Output videos
```

---

## Installation

### 1. Clone the repository

```
git clone https://github.com/<your-username>/person-reid-app.git
cd person-reid-app
```

### 2. Create a virtual environment

Using Conda:

```
conda create -n reid python=3.10 -y
conda activate reid
```

Using venv:

```
python -m venv reid_env
reid_env\Scripts\activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. (Important) Pull LFS model files

```
git lfs install
git lfs pull
```

---

## Running the Application

```
streamlit run app.py
```

Open the browser link displayed in the terminal.
Upload any video and click “Run Person Re-ID” to generate the processed output.

---

## How the System Works

1. YOLOv8-Seg detects and tracks people across frames.
2. For each tracked person:

   * Silhouette → Gait CNN-LSTM (256-dim embedding)
   * RGB crop → OSNet (512-dim embedding)
   * Keypoints → Body ratio features (2-dim)
3. The three feature vectors are concatenated into a 770-dim vector.
4. This vector is matched against an enrolled gallery to identify the person.

---

## Gallery Enrollment

The gallery file (`my_known_gallery.pth`) contains fused embeddings for known individuals. Enrollment is done using a separate script (from your training notebook).

---

## Notes

* The Gait model file is large (274 MB) and handled using Git LFS.
* CPU inference will be slow; GPU is recommended for real-time use.

---

# 1. **Architecture Diagram (ASCII)**

```text
                     +----------------------+
                     |     Streamlit UI     |
                     |  (Video Upload Page) |
                     +----------+-----------+
                                |
                                v
                   +---------------------------+
                   |     Inference Pipeline    |
                   |       (app.py → reid)     |
                   +---------------------------+
                                |
              ------------------------------------------------
              |                     |                       |
              v                     v                       v
     +----------------+   +--------------------+   +------------------+
     | YOLOv8-Seg     |   | YOLOv8-Pose       |   | OSNet (Torchreid)|
     | Detection &    |   | Keypoints (Body   |   | Appearance Embd. |
     | Tracking       |   | Ratios)           |   | (512-dim)        |
     +-------+--------+   +---------+----------+   +--------+---------+
             |                        |                      |
             |                        |                      |
             v                        v                      v
  +------------------+      +------------------+     +-------------------+
  | Silhouette Mask  |      | Body Ratio Vec   |     | RGB Crop → Embd.  |
  +--------+---------+      +--------+---------+     +---------+---------+
           |                         |                         |
           v                         |                         |
  +-----------------------+          |                         |
  | Gait CNN-LSTM Model   |          |                         |
  | (256-dim embedding)   |          |                         |
  +-----------+-----------+          |                         |
              \                     /                          |
               \                   /                           |
                \                 /                            |
                 v               v                             v
                      +------------------------------+
                      |   3-Way Feature Fusion       |
                      |  [256 + 512 + 2 = 770 dims]   |
                      +---------------+--------------+
                                      |
                                      v
                      +------------------------------+
                      | Gallery Matching (CDIST)     |
                      | Threshold-Based Re-ID        |
                      +---------------+--------------+
                                      |
                                      v
                      +------------------------------+
                      |  Person ID Assignment        |
                      |  (Known / Unknown)           |
                      +---------------+--------------+
                                      |
                                      v
                      +------------------------------+
                      | Final Video Rendering        |
                      | (Boxes + IDs drawn)          |
                      +------------------------------+
```

---

# 2. **Small Pipeline Illustration**

```text
Input Video
     ↓
YOLOv8-Seg → Person Tracking + Masks
     ↓
For each person:
     ├── Silhouette → Gait CNN-LSTM → 256-dim
     ├── RGB Crop → OSNet → 512-dim
     └── Keypoints → Body Ratios → 2-dim

Fuse Features → 770-dimensional vector
     ↓
Compare with Gallery Embeddings
     ↓
Assign Identity (Known / Unknown)
     ↓
Draw Bounding Boxes + Labels
     ↓
Output Processed Video
```

---


```

#### **Parameters**

| Name                | Type  | Description                                      |
| ------------------- | ----- | ------------------------------------------------ |
| `input_video_path`  | `str` | Path to the input video file.                    |
| `output_video_path` | `str` | Path where processed output video will be saved. |

---

### **Model Components Loaded internally**

| Component          | Description                              |
| ------------------ | ---------------------------------------- |
| YOLOv8-Seg         | Tracking, bounding boxes, silhouettes    |
| YOLOv8-Pose        | Keypoints for biometric ratios           |
| Gait CNN-LSTM      | Silhouette-based gait embedding          |
| OSNet (Torchreid)  | RGB crop appearance embedding            |
| Gallery embeddings | Stored in `gallery/my_known_gallery.pth` |

---

### **Processing Flow**

1. Load all models (lazy-loading cached)
2. Open input video with OpenCV
3. For each frame:

   * Run YOLOv8-Seg tracking
   * Extract person masks, crops, and IDs
   * Process gait sequence (when 30 frames are collected)
   * Process appearance embedding
   * Process body ratios
   * Compute final 770-dim embedding
   * Match with gallery using `torch.cdist`
   * Assign identity
   * Draw bounding box + label
4. Save output video

---
