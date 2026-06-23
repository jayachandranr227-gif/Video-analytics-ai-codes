import os

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from config import DATASET_DIR, FACE_MATCH_THRESHOLD

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1)

person_embeddings = {}


def _normalize(embedding):
    return embedding / np.linalg.norm(embedding)


def _largest_face(faces):
    return max(
        faces,
        key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]),
    )


def load_registered_faces():
    if not os.path.isdir(DATASET_DIR):
        print(f"Dataset folder not found: {DATASET_DIR}")
        return

    for person in os.listdir(DATASET_DIR):
        person_path = os.path.join(DATASET_DIR, person)
        if not os.path.isdir(person_path):
            continue

        embeddings = []

        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            image = cv2.imread(image_path)
            if image is None:
                continue

            faces = app.get(image)
            if not faces:
                continue

            face = _largest_face(faces)
            embeddings.append(_normalize(face.embedding))

        if embeddings:
            mean_embedding = np.mean(embeddings, axis=0)
            person_embeddings[person] = _normalize(mean_embedding)

    print(f"Loaded {len(person_embeddings)} registered people")


def recognize(face_image):
    faces = app.get(face_image)
    if not faces:
        return "UNKNOWN", 0.0

    face = _largest_face(faces)
    embedding = _normalize(face.embedding)

    best_name = "UNKNOWN"
    best_score = -1.0

    for name, known_embedding in person_embeddings.items():
        score = float(np.dot(embedding, known_embedding))
        if score > best_score:
            best_score = score
            best_name = name

    if best_score < FACE_MATCH_THRESHOLD:
        return "UNKNOWN", best_score

    return best_name, best_score


load_registered_faces()
