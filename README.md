# 🧠 Face Recognition Attendance System

A **real-time, AI-powered attendance system** that uses **InsightFace** for face detection and recognition, enabling seamless and automated attendance marking.
It supports **live camera feeds**, **video processing**, **email notifications**, and **SQLite + CSV storage** — optimized for both **CPU and GPU (CUDA)**.

---

## 🚀 Features

✅ **Real-Time Face Detection & Recognition**
Uses **InsightFace (buffalo_l)** model for robust detection and recognition.

✅ **GPU Acceleration**
Auto-detects and leverages **CUDA** via **ONNXRuntime GPU** if available.

✅ **Automatic Attendance Marking**
Detects and identifies known faces in live video or recorded footage.

✅ **Database + CSV Storage**
Stores all attendance data in **SQLite** and appends CSV logs automatically.

✅ **Instant Email Notifications**
Sends real-time and daily summary attendance reports via SMTP.

✅ **Dataset Enrollment Utility**
Enroll new faces from folder-based datasets with one command.

✅ **Simple & Modular Codebase**
Each component — detection, recognition, DB, tracker, and notifier — is cleanly separated for easy maintenance.

---

## 🧩 Project Architecture

```
attendance_system/
├── attendance_system.py     # Main real-time attendance logic
├── enroll_dataset.py        # Bulk face enrollment from dataset folders
├── face_detection.py        # Face detection using InsightFace (RetinaFace)
├── face_recognition.py      # Face recognition (ArcFace embeddings)
├── db_writer.py             # Async attendance writer (DB + CSV + Email)
├── email_notification.py    # Email report generation and sending
├── simple_tracker.py        # Lightweight IOU-based tracker
├── config.py                # All configuration and constants
├── test_video.py            # Test attendance from pre-recorded video
├── main.py                  # Entry point with CLI options
├── requirements.txt         # Python dependencies
└── environment.yml          # Conda environment configuration
```

---

## ⚙️ Installation

### 1️⃣ Create Environment

#### Option A: Using Conda

```bash
conda env create -f environment.yml
conda activate attendance-env
```

#### Option B: Using Pip

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

### 2️⃣ Configure Settings

Open **`config.py`** and update your email credentials:

```python
EMAIL_ADDRESS = "your_email@gmail.com"
EMAIL_PASSWORD = "your_app_password"
```

> ⚠️ **Important:**
> For Gmail, enable **2FA** and use a generated **App Password** — not your regular email password.

You can also adjust:

* `FACE_RECOGNITION_THRESHOLD` — controls recognition sensitivity
* `ATTENDANCE_COOLDOWN` — seconds before the same person can be re-marked
* `CAMERA_INDEX` — use different camera sources

---

## 🧑‍💼 Usage

### ▶️ Run the Real-Time Attendance System

```bash
python main.py
```

> Press **`q`** in the camera window to quit.

---

### 📚 Enroll New Faces from Dataset

Prepare a dataset like:

```
datasets/
├── Alice/
│   ├── img1.jpg
│   └── img2.jpg
└── Bob/
    ├── img1.jpg
    └── img2.jpg
```

Then run:

```bash
python main.py --enroll ./datasets
```

---

### 🎞️ Process a Recorded Video

To analyze a saved video file and mark attendance automatically:

```bash
python main.py --test-video ./videos/meeting.mp4
```

Annotated output and attendance CSV will be saved in the `attendance_reports/` folder.

---

### 📧 Test Email Configuration

```bash
python main.py --test-email
```

---

## 🗄️ Data Storage

**SQLite Database (`attendance.db`)**

* `registered_faces` — enrolled individuals and embeddings
* `attendance` — daily attendance records

**CSV Reports (`attendance_reports/`)**

* Logs every attendance mark
* Includes **PersonID, Name, Date, Time, Confidence**

---

## ⚡ Performance Tips

* Use a **GPU** if available (`onnxruntime-gpu` will be used automatically).
* Increase `RECOG_PERIOD` in `attendance_system.py` for higher FPS.
* Add multiple face samples per person for better recognition accuracy.

---

## 🔒 Security Recommendations

* Never hardcode passwords — use **environment variables**.
* Restrict access to `attendance.db` and CSV files.
* Regularly backup the database and reports.

---

## 🧠 Technologies Used

| Component                    | Library                                                   |
| ---------------------------- | --------------------------------------------------------- |
| Face Detection & Recognition | [InsightFace](https://github.com/deepinsight/insightface) |
| Model Runtime                | ONNXRuntime (GPU/CPU)                                     |
| Tracking                     | Custom IOU-based tracker                                  |
| Storage                      | SQLite + CSV                                              |
| Notifications                | smtplib (Email)                                           |
| Visualization                | OpenCV                                                    |
| Embedding Management         | NumPy, Pickle                                             |

---

## 🧪 Testing

The `testing/` folder includes quick validation scripts:

```bash
python testing/test_insightface_models.py  # Downloads and validates models
python testing/test_insightface.py         # Verifies face detection pipeline
```

---

## 📝 License

This project is licensed under the **MIT License** — free for personal and commercial use.
