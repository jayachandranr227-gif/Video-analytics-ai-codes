import csv
import os
from datetime import datetime

from config import ATTENDANCE_LOG, LOG_DIR

marked_today = set()


def mark(name, camera_name):
    today = datetime.now().strftime("%Y-%m-%d")
    key = (today, name, camera_name)

    if key in marked_today:
        return

    marked_today.add(key)
    os.makedirs(LOG_DIR, exist_ok=True)

    file_exists = os.path.exists(ATTENDANCE_LOG)

    with open(ATTENDANCE_LOG, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(["date", "time", "name", "camera"])

        now = datetime.now()
        writer.writerow([
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            name,
            camera_name,
        ])

    print(f"Attendance marked: {name} from {camera_name}")
