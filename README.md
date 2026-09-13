# 🔍 Behavioral Analysis

> **Computer vision** for real-time analysis of human behavior.

---

## 🪧 About

This project brings together **two separate** computer vision applications that analyze people's behavior from a webcam or a recorded video. They do not share the same detection engine:

| | `app.py` (live webcam) | `detect.py` (recorded video) |
| :--- | :--- | :--- |
| **Detection** | MediaPipe Holistic (pose + face + hands) | YOLOv8-pose (`ultralytics`) with ByteTrack tracking |
| **What is measured** | Gaze (attentive/distracted), posture (good/bad), duration | Posture (fuzzy), arm position, restlessness (wrist speed), facial emotion (DeepFace), estimated psychological state |
| **Export** | CSV statistics | — (no export for now) |

**Why this project?**
- Detect posture to help prevent back problems
- Analyze gaze and restlessness to measure attention/stress
- Explore MediaPipe, YOLO-pose, DeepFace, and fuzzy logic (`scikit-fuzzy`) for behavioral classification
- Build a simple, interactive interface with Streamlit

⚠️ **Note**: the two scripts are independent — this isn't a "mode 1 / mode 2" of the same engine, but two different prototypes within the same repository.

---
## 📸 Screenshots

![image alt](https://github.com/abidirymeh/analyse-comportementale/blob/57455a820ab7bb3c7de7d5fc6a6a2d33e264d8b9/capture1.jfif)
![image alt](https://github.com/abidirymeh/analyse-comportementale/blob/57455a820ab7bb3c7de7d5fc6a6a2d33e264d8b9/capture2.jfif)
![image alt](https://github.com/abidirymeh/analyse-comportementale/blob/57455a820ab7bb3c7de7d5fc6a6a2d33e264d8b9/capture3.jfif)
![image alt](https://github.com/abidirymeh/analyse-comportementale/blob/57455a820ab7bb3c7de7d5fc6a6a2d33e264d8b9/capture4.jfif)
![image alt](https://github.com/abidirymeh/analyse-comportementale/blob/57455a820ab7bb3c7de7d5fc6a6a2d33e264d8b9/capture5.jfif)

---

## 📦 Requirements

| Item | Minimum version | Link |
| :--- | :--- | :--- |
| **Python** | 3.10+ | [python.org](https://python.org) |
| **pip** | 22.0+ | [pip documentation](https://pip.pypa.io/) |
| **Git** | 2.0+ | [git-scm.com](https://git-scm.com/) |

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/abidirymeh/analyse-comportementale.git
cd analyse-comportementale
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

**Windows:**
```bash
venv\Scripts\activate
```

**Mac / Linux:**
```bash
source venv/bin/activate
```

### 4. Install the dependencies

```bash
pip install -r requirements.txt
```

### 5. Check the installation

```bash
pip list
```

###
## 🛠️ Usage

### Mode 1: Live webcam — `app.py` (MediaPipe)

```bash
streamlit run app.py
```

Then open your browser at: http://localhost:8501

Features:
- ✅ Real-time capture and analysis via WebRTC
- ✅ Gaze detection (attentive / distracted) and posture
- ✅ Interactive dashboard in the sidebar
- ✅ Export statistics to CSV

### Mode 2: Recorded video — `detect.py` (YOLO-pose + DeepFace + fuzzy logic)

```bash
streamlit run detect.py
```

In the sidebar, configure:
- **Video path**
- **Playback speed** and **frames to skip**
- Display of confidence and skeleton (with adjustable thickness)

Then click **▶️ Start** (and **⏹️ Stop** to stop).

Features:
- ✅ Multi-person tracking (ByteTrack)
- ✅ Fuzzy classification of posture (standing / sitting / leaning)
- ✅ Arm position detection (crossed / open / hidden)
- ✅ Restlessness estimation (wrist speed)
- ✅ Facial emotion recognition (DeepFace)
- ✅ Estimation of a "psychological state" combining these signals (e.g. 😊 Confident, 😰 Stressed/Anxious)

### Mode 3: Export data (webcam only)

Once the `app.py` analysis is running, click **"Export CSV"** in the sidebar.

CSV file format:
```csv
ID,Attention_%,Bad_posture_%,Duration_s
1,95.2,5.8,12
2,67.3,32.7,8
```

---

## 🤝 Contributing

Contributions are welcome!

1. **Fork the project** — the "Fork" button at the top right of the GitHub repository.
2. **Clone your fork**
   ```bash
   git clone https://github.com/abidirymeh/analyse-comportementale.git
   cd analyse-comportementale
   ```
3. **Create a branch for your feature**
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make your changes** — add your code, make sure everything works, update the docs if needed.
5. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add feature X"
   ```
6. **Push your branch**
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Open a Pull Request** against the `main` branch of the original repository.

**Commit conventions:**
- `feat:` for a new feature
- `fix:` for a bug fix
- `docs:` for documentation
- `style:` for formatting
- `refactor:` for refactoring
- `test:` for tests
- `chore:` for maintenance tasks

---

## 🏗️ Built With

### Languages & Frameworks

| Tool | Role | Link |
| :--- | :--- | :--- |
| Python 3.10+ | Main language | [python.org](https://python.org) |
| Streamlit | User interface | [streamlit.io](https://streamlit.io) |
| Streamlit-WebRTC | Webcam capture (`app.py`) | [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc) |
| MediaPipe | Pose/face/hand detection (`app.py`) | [mediapipe.dev](https://mediapipe.dev) |
| Ultralytics YOLOv8-pose | Pose detection + tracking (`detect.py`) | [ultralytics.com](https://ultralytics.com) |
| DeepFace | Facial emotion recognition (`detect.py`) | [github.com/serengil/deepface](https://github.com/serengil/deepface) |
| scikit-fuzzy | Fuzzy classification of posture/arms/restlessness (`detect.py`) | [pypi.org/project/scikit-fuzzy](https://pypi.org/project/scikit-fuzzy) |
| OpenCV | Image processing | [opencv.org](https://opencv.org) |
| NumPy | Numerical computation | [numpy.org](https://numpy.org) |
| Pandas | Data handling | [pandas.pydata.org](https://pandas.pydata.org) |

### Tools

| Tool | Role | Link |
| :--- | :--- | :--- |
| Git | Version control | [git-scm.com](https://git-scm.com) |
| GitHub | Code hosting | [github.com](https://github.com) |
| VSCode | Code editor | [code.visualstudio.com](https://code.visualstudio.com) |

### CI / CD

This project does not currently use continuous integration. A future improvement to consider.

### Deployment

Designed to run locally. Online deployment is conceivable with Streamlit Cloud or Hugging Face Spaces — provided the compatibility of YOLO/DeepFace with the target environment is checked (model weights, memory, GPU).

| Platform | Link |
| :--- | :--- |
| Streamlit Cloud | [streamlit.io/cloud](https://streamlit.io/cloud) |
| Hugging Face Spaces | [huggingface.co/spaces](https://huggingface.co/spaces) |

---

## 📚 Documentation

### Embedded documentation
- `README.md` — general overview of the project
- `requirements.txt` — list of dependencies

### External documentation

| Topic | Link |
| :--- | :--- |
| MediaPipe Pose | [MediaPipe Pose Guide](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) |
| Ultralytics YOLO Pose | [docs.ultralytics.com](https://docs.ultralytics.com/tasks/pose/) |
| DeepFace | [github.com/serengil/deepface](https://github.com/serengil/deepface) |
| scikit-fuzzy | [pythonhosted.org/scikit-fuzzy](https://pythonhosted.org/scikit-fuzzy/) |
| Streamlit Documentation | [docs.streamlit.io](https://docs.streamlit.io) |
| OpenCV Tutorials | [docs.opencv.org](https://docs.opencv.org) |

### Code structure

```
CV2/
│
├── core/                          # Modules for app.py (to be created, see Installation §6)
│   ├── detectors.py               # HolisticDetector (MediaPipe)
│   ├── tracker.py                 # PersonTracker
│   ├── analyzers.py               # AttentionAnalyzer, PostureAnalyzer
│   └── visualizer.py              # Visualizer (OpenCV overlays)
│
├                     
├── RESULT.mp4                     # Example video for detect.py
│
├── app.py                         # Live webcam app (MediaPipe)
├── src/
│     └── detect.py                # Video app (YOLO-pose + DeepFace + fuzzy logic)
│                         
├── requirements.txt               # Dependencies
├── README.md                      # Documentation
└── .gitignore                     # Ignored files
```

---

## 🏷️ Versioning

Version naming follows [Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`

- **MAJOR**: changes incompatible with earlier versions
- **MINOR**: added features (backward-compatible)
- **PATCH**: bug fixes (backward-compatible)

Versions and changelogs are available on the Releases page.

---

## 📝 License

See the repository's `LICENSE` file. This project is licensed under the MIT License — you may use, modify, and distribute it freely.

---

## 📧 Contact

- **Author**: [Rimeh Abidi]
- **Email**: [rimeh.abidi@enis.tn]
- **GitHub**: https://github.com/abidirymeh
