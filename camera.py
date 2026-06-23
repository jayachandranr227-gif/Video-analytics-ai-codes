import cv2


def start_camera(source):
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera source: {source}")

    return cap
