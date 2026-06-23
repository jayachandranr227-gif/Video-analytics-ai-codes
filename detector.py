from ultralytics import YOLO

from config import YOLO_MODEL_PATH

model = YOLO(YOLO_MODEL_PATH)


def detect(frame):
    return model(frame, verbose=False)
