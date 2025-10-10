## 📸 Face Biometric Attendance System

A robust, real-time solution for **automatic attendance marking** using advanced face detection and recognition. It leverages **YuNet** for fast face detection and **FaceNet** for accurate identification, complete with database storage, email reports, and an intuitive interface.

---

## ✨ Key Features

* **Real-time Processing**: Instantly detects faces using the **YuNet** model.
* **Accurate Recognition**: Identifies personnel using **FaceNet** to match against known faces.
* **Automatic Marking**: Records attendance instantly upon successful face recognition.
* **Email Notifications**: Sends attendance records as **CSV** reports via email.
* **Persistent Storage**: Uses an **SQLite database** to securely store attendance history and face encodings.
* **Interactive Interface**: Simple-to-use controls for managing the system and enrolling new personnel.

---

## 💻 Prerequisites

To run the system, you'll need:

* **Python** (3.7+)
* **OpenCV** (4.8+)
* **TensorFlow** (2.13+)
* A working **Camera**
* **Email Configuration** (Gmail is recommended; requires an **App Password**)

---

## ⚙️ Installation

1.  **Get the Code**: Clone or download the project files.
2.  **Run Setup**: Execute the setup script:
    ```bash
    python setup.py
    ```
3.  **Manual Dependencies (Optional)**: If the setup fails, install dependencies directly:
    ```bash
    pip install -r requirements.txt
    ```

---

## 🔒 Configuration

1.  **Update Email Settings**: Edit `config.py` with your email details:
    ```python
    EMAIL_ADDRESS = "your_email@gmail.com"
    EMAIL_PASSWORD = "your_app_password"  # Important: Use a generated App Password for Gmail
    ```
2.  **Gmail Security**: If using Gmail, you must enable **2-factor authentication** and then **generate an app password** for the system to send emails.

---

## ▶️ Usage

### Starting the System

To launch the real-time attendance system:

```bash
python main.py
````

### Command-Line Utilities

Utilize optional arguments for specific tasks:

| Command | Description |
| :--- | :--- |
| `python main.py --test-email` | Verify your email configuration |
| `python main.py --send-report` | Manually trigger an attendance report |
| `python main.py --stats` | Display attendance statistics |
| `python main.py --camera 1` | Specify a different camera index (e.g., 1) |

### Interactive Controls (While Running)

Control the system directly within the camera view:

  * Press **'q'** to **Quit** the application.
  * Press **'a'** to **Add a new person**.
  * Press **'s'** to **Send a manual attendance report**.

-----

## 🧑‍🎓 Adding New Personnel

1.  Run the main system (`python main.py`).
2.  Position the new person's face clearly in front of the camera.
3.  Press the **'a'** key.
4.  Enter the required **ID** and **Name** when prompted.

The system will automatically capture the face and store its unique **FaceNet encoding** in the database.

-----

## 📧 Email Reports

Attendance reports are sent in **CSV format** and include: **Person ID**, **Person Name**, **Timestamp**, and **Confidence Score**.

  * **Immediate Notifications**: Sent when an individual is successfully marked present.
  * **Scheduled Reports**: Sent automatically at configurable intervals.
  * **Manual Reports**: Can be triggered anytime using the `'s'` key or `--send-report` option.

-----

## 🗄️ Database Structure (SQLite)

### Attendance Table

Records of every check-in:

  * `id` (Primary Key)
  * `person_id`
  * `person_name`
  * `timestamp`
  * `confidence` (Recognition score)

### Known Faces Table

Stores identification data:

  * `id` (Primary Key)
  * `person_id` (Unique identifier for the person)
  * `person_name`
  * `face_encoding` (The stored FaceNet vector)
  * `created_at`

-----

## ❓ Troubleshooting

| Issue | Solution |
| :--- | :--- |
| **Camera** | Check connection, camera permissions, and try different indices (`--camera 0`, `1`, etc.). |
| **Email** | Double-check `config.py` credentials, ensure you are using a **Gmail app password**, and verify SMTP settings. |
| **Models** | Confirm internet access for the initial download, disk space, and file permissions. |
| **Recognition** | Ensure bright, even lighting, a clear view of the face, and consider adding multiple face samples for better accuracy. |

-----

## 📁 Project Files

```
├── main.py                 # Application entry point
├── attendance_system.py    # Core logic and flow
├── face_detection.py       # YuNet implementation
├── face_recognition.py     # FaceNet implementation
├── database.py            # DB handling (SQLite)
├── email_notification.py  # Email sending logic
├── config.py              # Settings and credentials
├── setup.py               # Setup script
└── requirements.txt       # Python dependencies
```

-----

## ⚡ Performance & Security

### Performance

  * Includes a **cooldown period** to prevent redundant attendance marks.
  * The face recognition **confidence threshold** can be adjusted in `config.py`.
  * Database operations are optimized for speed in a real-time environment.

### Security

  * Face encodings are stored as **non-reversible binary data**.
  * It's highly recommended to use **environment variables** for storing sensitive email credentials instead of directly in `config.py`.
  * Ensure regular **database backups**.

-----

## 📝 License

This project is freely available under the **MIT License**.

