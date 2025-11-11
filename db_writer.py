# db_writer.py
import threading
import queue
import pandas as pd
import os
from datetime import datetime
from email_notification import EmailNotifier

class DBWriter:
    """
    Background writer to persist attendance records to CSV files
    and trigger immediate email notifications.
    Attendance records are stored only in CSV files, not in the database.
    """
    def __init__(self, csv_path):
        self.q = queue.Queue()
        self._stop_event = threading.Event()
        self.notifier = EmailNotifier()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        
        # Store the unique CSV path for this session
        self.csv_path = os.path.abspath(csv_path)  # Use absolute path
        csv_dir = os.path.dirname(self.csv_path)
        if csv_dir:  # Only create directory if path has a directory component
            os.makedirs(csv_dir, exist_ok=True)
        print(f"📁 CSV file will be saved to: {self.csv_path}")
        self.thread.start()

    def enqueue(self, person_id, person_name, timestamp_str, confidence):
        """Add a new attendance record to the background queue."""
        self.q.put((person_id, person_name, timestamp_str, confidence))
        print(f"📥 Enqueued attendance: {person_name} (ID: {person_id}) @ {timestamp_str}")

    def _worker(self):
        """Worker loop that writes attendance records to CSV files asynchronously."""
        while not self._stop_event.is_set():
            try:
                # Get item from queue with timeout
                item = self.q.get(timeout=1.0)
            except queue.Empty:
                # Timeout is normal, just continue the loop
                continue
            except Exception as e:
                print(f"⚠️ DBWriter queue error: {e}")
                continue

            if item is None:
                break

            pid, name, ts, conf = item
            try:
                # --- Write to Session CSV (Attendance records stored only in CSV files) ---
                date_str, time_str = ts.split(' ')
                df = pd.DataFrame([[pid, name, date_str, time_str, float(conf)]],
                                  columns=["PersonID", "Name", "Date", "Time", "Confidence"])
                
                # Ensure directory exists
                csv_dir = os.path.dirname(self.csv_path)
                if csv_dir:
                    os.makedirs(csv_dir, exist_ok=True)
                
                # Check if file exists before writing
                file_exists = os.path.exists(self.csv_path)
                
                # Write to CSV with explicit encoding and line terminator
                df.to_csv(
                    self.csv_path, 
                    mode='a', 
                    header=not file_exists, 
                    index=False,
                    encoding='utf-8',
                    lineterminator='\n'
                )
                
                # Verify file was created/updated
                if os.path.exists(self.csv_path):
                    print(f"✅ Attendance recorded: {name} @ {date_str} {time_str} -> {self.csv_path}")
                else:
                    print(f"❌ ERROR: CSV file was not created at {self.csv_path}")

                # --- Send Email Notification ---
                try:
                    self.notifier.send_immediate_notification(
                        person_name=name,
                        timestamp=f"{date_str} {time_str}"
                    )
                except Exception as email_error:
                    print(f"⚠️ Email notification error: {email_error}")

            except Exception as e:
                import traceback
                print(f"⚠️ DBWriter error writing CSV: {e}")
                print(f"   CSV path: {self.csv_path}")
                traceback.print_exc()
            finally:
                try:
                    self.q.task_done()
                except Exception:
                    pass

    def flush(self):
        """Wait for all pending items in the queue to be processed."""
        self.q.join()  # Wait for all tasks to be done
    
    def stop(self):
        """Stop the background thread gracefully."""
        # Wait for all pending items to be processed
        self.q.join()
        # Signal the thread to stop
        self._stop_event.set()
        self.q.put(None)
        self.thread.join(timeout=5.0)  # Wait up to 5 seconds for thread to finish
        if self.thread.is_alive():
            print("⚠️ Warning: DBWriter thread did not stop gracefully")