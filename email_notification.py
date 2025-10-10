import smtplib
import pandas as pd
import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os
from config import SMTP_SERVER, SMTP_PORT, EMAIL_ADDRESS, EMAIL_PASSWORD, CSV_FILENAME

class EmailNotifier:
    def __init__(self):
        self.smtp_server = SMTP_SERVER
        self.smtp_port = SMTP_PORT
        self.email_address = EMAIL_ADDRESS
        self.email_password = EMAIL_PASSWORD
        self.csv_filename = CSV_FILENAME
    
    def create_csv_report(self, attendance_records, recipient_email="admin@company.com"):
        """Create CSV report from attendance records"""
        if not attendance_records:
            return None
        
        # Convert records to DataFrame
        df = pd.DataFrame(attendance_records, columns=[
            'Person ID', 'Person Name', 'Timestamp', 'Confidence'
        ])
        
        # Format timestamp
        df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Add summary statistics
        summary = {
            'Total Records': len(df),
            'Unique People': df['Person Name'].nunique(),
            'Date Range': f"{df['Timestamp'].min()} to {df['Timestamp'].max()}",
            'Generated At': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save CSV file
        csv_path = f"reports/{self.csv_filename}"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        
        return csv_path, summary
    
    def send_attendance_report(self, attendance_records, recipient_email="admin@company.com"):
        """Send attendance report via email"""
        try:
            # Create CSV report
            csv_path, summary = self.create_csv_report(attendance_records, recipient_email)
            if not csv_path:
                print("No attendance records to send")
                return False
            
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = self.email_address
            msg['To'] = recipient_email
            msg['Subject'] = f"Attendance Report - {datetime.datetime.now().strftime('%Y-%m-%d')}"
            
            # Email body
            body = f"""
            <html>
            <body>
                <h2>Daily Attendance Report</h2>
                <p><strong>Report Summary:</strong></p>
                <ul>
                    <li>Total Records: {summary['Total Records']}</li>
                    <li>Unique People: {summary['Unique People']}</li>
                    <li>Date Range: {summary['Date Range']}</li>
                    <li>Generated At: {summary['Generated At']}</li>
                </ul>
                <p>Please find the detailed attendance records in the attached CSV file.</p>
                <p>Best regards,<br>Attendance System</p>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            # Attach CSV file
            with open(csv_path, "rb") as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
            
            encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename= {os.path.basename(csv_path)}'
            )
            msg.attach(part)
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email_address, self.email_password)
            text = msg.as_string()
            server.sendmail(self.email_address, recipient_email, text)
            server.quit()
            
            print(f"Attendance report sent successfully to {recipient_email}")
            return True
            
        except Exception as e:
            print(f"Error sending email: {e}")
            return False
    
    def send_immediate_notification(self, person_name, timestamp, recipient_email="admin@company.com"):
        """Send immediate notification when someone is marked present"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_address
            msg['To'] = recipient_email
            msg['Subject'] = f"Attendance Marked - {person_name}"
            
            body = f"""
            <html>
            <body>
                <h3>Attendance Notification</h3>
                <p><strong>Person:</strong> {person_name}</p>
                <p><strong>Time:</strong> {timestamp}</p>
                <p>This person has been automatically marked present in the attendance system.</p>
                <p>Best regards,<br>Attendance System</p>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email_address, self.email_password)
            text = msg.as_string()
            server.sendmail(self.email_address, recipient_email, text)
            server.quit()
            
            print(f"Immediate notification sent for {person_name}")
            return True
            
        except Exception as e:
            print(f"Error sending immediate notification: {e}")
            return False
    
    def test_email_connection(self):
        """Test email connection"""
        try:
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email_address, self.email_password)
            server.quit()
            print("Email connection test successful")
            return True
        except Exception as e:
            print(f"Email connection test failed: {e}")
            return False
