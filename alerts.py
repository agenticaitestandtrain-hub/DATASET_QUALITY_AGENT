import smtplib
from email.mime.text import MIMEText
import smtplib
from email.message import EmailMessage


def send_alert(filename, report_file):

    sender = "agenticaitestandtrain@gmail.com"
    password = "pgvnlnlnanhogbtr"
    receiver = "agenticaitestandtrain@gmail.com"

    # create email object
    msg = EmailMessage()

    msg["Subject"] = "Dataset Analysis Report"
    msg["From"] = sender
    msg["To"] = receiver

    msg.set_content(f"A dataset was uploaded: {filename}\nFull analysis report attached.")

    # attach report file
    with open(report_file, "rb") as f:
        file_data = f.read()
        file_name = report_file

    msg.add_attachment(
        file_data,
        maintype="application",
        subtype="octet-stream",
        filename=file_name
    )

    # send email
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(sender, password)
        smtp.send_message(msg)