import os
from collections import Counter, defaultdict, deque
from datetime import datetime

import cv2

from attendance import mark
from camera import start_camera
from config import (
    CAMERA_SOURCES,
    HISTORY_SIZE,
    MIN_KNOWN_FRAMES,
    PERSON_CONFIDENCE,
    UNAUTHORIZED_DIR,
    UNAUTHORIZED_FRAMES,
)
from detector import detect
from liveness import check_liveness
from recognizer import recognize

prediction_history = defaultdict(lambda: deque(maxlen=HISTORY_SIZE))
last_unauthorized_save = {}


def choose_display_label(camera_name, name, score):
    history = prediction_history[camera_name]
    history.append((name, score))

    known_predictions = [item for item in history if item[0] != "UNKNOWN"]
    if len(known_predictions) >= MIN_KNOWN_FRAMES:
        counts = Counter(item[0] for item in known_predictions)
        best_name, count = counts.most_common(1)[0]

        if count >= MIN_KNOWN_FRAMES:
            matching_scores = [item[1] for item in known_predictions if item[0] == best_name]
            average_score = sum(matching_scores) / len(matching_scores)
            return best_name, average_score, (0, 255, 0)

    if len(history) == HISTORY_SIZE and all(item[0] == "UNKNOWN" for item in history):
        return "UNAUTHORIZED", 0.0, (0, 0, 255)

    return "SCANNING", 0.0, (0, 255, 255)


def save_unauthorized(camera_name, frame):
    now = datetime.now()
    last_saved = last_unauthorized_save.get(camera_name)

    if last_saved and (now - last_saved).total_seconds() < UNAUTHORIZED_FRAMES:
        return

    last_unauthorized_save[camera_name] = now
    os.makedirs(UNAUTHORIZED_DIR, exist_ok=True)

    filename = f"{camera_name}_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
    path = os.path.join(UNAUTHORIZED_DIR, filename)
    cv2.imwrite(path, frame)
    print(f"Unauthorized saved: {path}")


def process_frame(camera_name, frame):
    results = detect(frame)

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if class_id != 0 or confidence < PERSON_CONFIDENCE:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                continue

            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            head_height = max(1, int(person_crop.shape[0] * 0.6))
            head_crop = person_crop[:head_height, :]

            name, score = recognize(head_crop)

            if name != "UNKNOWN" and check_liveness(head_crop):
                display_name, display_score, color = choose_display_label(camera_name, name, score)
                if display_name not in ("SCANNING", "UNAUTHORIZED"):
                    mark(display_name, camera_name)
            else:
                display_name, display_score, color = choose_display_label(camera_name, "UNKNOWN", 0.0)

            if display_name == "SCANNING":
                label = "SCANNING"
            elif display_name == "UNAUTHORIZED":
                label = "UNAUTHORIZED"
                save_unauthorized(camera_name, person_crop)
            else:
                label = f"{display_name} {display_score:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

    return frame


def main():
    cameras = {}

    for camera_name, source in CAMERA_SOURCES.items():
        cameras[camera_name] = start_camera(source)
        print(f"Camera started: {camera_name}")

    try:
        while True:
            for camera_name, cap in cameras.items():
                ret, frame = cap.read()
                if not ret:
                    print(f"Frame not received from {camera_name}")
                    continue

                output = process_frame(camera_name, frame)
                cv2.imshow(f"Attendance - {camera_name}", output)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        for cap in cameras.values():
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
