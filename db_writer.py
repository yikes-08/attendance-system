# db_writer.py
import threading
import queue
import sqlite3
import pandas as pd
import os
from datetime import datetime
from config import DATABASE_PATH, CSV_FILENAME
from email_notification import EmailNotifier

class DBWriter:
    """
    Background writer to persist attendance records to DB + CSV
    and trigger immediate email notifications.
    """

    def __init__(self):
        self.q = queue.Queue()
        self._stop_event = threading.Event()
        self.notifier = EmailNotifier()  # ✅ initialize once
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def enqueue(self, person_id, person_name, timestamp_str, confidence):
        """Add a new attendance record to the background queue."""
        self.q.put((person_id, person_name, timestamp_str, confidence))

    def _worker(self):
        """Worker loop that writes to SQLite and CSV asynchronously."""
        while not self._stop_event.is_set():
            try:
                item = self.q.get(timeout=1.0)
            except Exception:
                continue

            if item is None:
                break

            pid, name, ts, conf = item
            try:
                # --- Ensure DB table exists ---
                conn = sqlite3.connect(DATABASE_PATH)
                cur = conn.cursor()
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS attendance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        person_id INTEGER,
                        name TEXT,
                        date TEXT,
                        time TEXT,
                        confidence REAL
                    )
                """)
                date_str, time_str = ts.split(' ')
                cur.execute(
                    "INSERT INTO attendance (person_id, name, date, time, confidence) VALUES (?, ?, ?, ?, ?)",
                    (pid, name, date_str, time_str, float(conf))
                )
                conn.commit()
                conn.close()

                # --- Write to CSV ---
                os.makedirs("attendance_reports", exist_ok=True)
                csv_path = os.path.join("attendance_reports", os.path.basename(CSV_FILENAME))
                df = pd.DataFrame([[pid, name, date_str, time_str, float(conf)]],
                                  columns=["PersonID", "Name", "Date", "Time", "Confidence"])
                file_exists = os.path.exists(csv_path)
                df.to_csv(csv_path, mode='a', header=not file_exists, index=False)

                # --- Send immediate email notification ---
                self.notifier.send_immediate_notification(
                    person_name=name,
                    timestamp=f"{date_str} {time_str}"
                )

            except Exception as e:
                print(f"⚠️ DBWriter error: {e}")
            finally:
                try:
                    self.q.task_done()
                except Exception:
                    pass

    def stop(self):
        """Stop the background thread gracefully."""
        self._stop_event.set()
        self.q.put(None)
        self.thread.join()
