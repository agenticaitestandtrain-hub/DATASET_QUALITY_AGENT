import smtplib
from email.mime.text import MIMEText
import smtplib
from email.message import EmailMessage


import requests

import requests

def send_alert(dataset, report):

    BOT_TOKEN = "8646525735:AAHGuoCitm6jrBpy0KFWkMJgMggy8C84OJQ"
    CHAT_ID = "5079531217"

    message = f"⚠ Dataset Alert\nDataset uploaded: {dataset}"

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    payload = {
        "chat_id": CHAT_ID,
        "text": message
    }

    requests.post(url, data=payload)