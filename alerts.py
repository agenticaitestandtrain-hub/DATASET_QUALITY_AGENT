import smtplib
from email.mime.text import MIMEText
import smtplib
from email.message import EmailMessage


import requests

def send_alert(dataset, report):

    BOT_TOKEN = "8674975796:AAHqPSM2HgHHIr1OtPaKhBUx4Xl96RnZWK8"
    CHAT_ID = "5079531217"

    message = f"⚠ Dataset Alert\nDataset: {dataset}"

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    payload = {
        "chat_id": CHAT_ID,
        "text": message
    }

    response = requests.post(url, data=payload)

    return response.status_code