import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

YOLO_MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
LOG_DIR = os.path.join(BASE_DIR, "logs")
ATTENDANCE_LOG = os.path.join(LOG_DIR, "attendance.csv")
UNAUTHORIZED_DIR = os.path.join(LOG_DIR, "unauthorized")

# Use 0 for laptop webcam testing.
# Replace/add RTSP URLs when you move to IP cameras.
CAMERA_SOURCES = {
    "webcam": 0,
    # "gate_1": "rtsp://username:password@192.168.1.10:554/stream1",
    # "gate_2": "rtsp://username:password@192.168.1.11:554/stream1",
}

PERSON_CONFIDENCE = 0.65
FACE_MATCH_THRESHOLD = 0.62
HISTORY_SIZE = 8
MIN_KNOWN_FRAMES = 3
UNAUTHORIZED_FRAMES = 8
