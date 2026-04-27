"""
bithuman_expression/utils/cpu_face_handler.py
Python replacement for cpu_face_handler.cpython-310-x86_64-linux-gnu.so

Exposes:
  CPUFaceHandler(model_selection=0)
      .detect(image: np.ndarray) -> (num_faces: int, bbox: List[int])
      .__call__(image)            -> same as detect()
"""

import numpy as np
from typing import List, Tuple


class CPUFaceHandler:
    """
    CPU-based face detector wrapping MediaPipe face_detection.

    Args:
        model_selection: 0 = short-range model (≤2 m, default),
                         1 = full-range model (≤5 m)
    """

    def __init__(self, model_selection: int = 0):
        import mediapipe as mp
        self._mp_face = mp.solutions.face_detection
        self._detector = self._mp_face.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=0.5,
        )

    def detect(self, image: np.ndarray) -> Tuple[int, List[int]]:
        """
        Detect faces in an RGB numpy image.

        Args:
            image: HxWx3 uint8 numpy array in RGB order.

        Returns:
            (num_faces, bbox)
            - num_faces: total detections found
            - bbox: [x1, y1, x2, y2] pixel coordinates if exactly one face
                    detected, empty list otherwise
        """
        h, w = image.shape[:2]
        results = self._detector.process(image)

        if not results.detections:
            return 0, []

        num_faces = len(results.detections)

        if num_faces != 1:
            return num_faces, []

        # Extract the single detection's bounding box
        bb = results.detections[0].location_data.relative_bounding_box
        x1 = max(0, int(bb.xmin * w))
        y1 = max(0, int(bb.ymin * h))
        x2 = min(w, int((bb.xmin + bb.width) * w))
        y2 = min(h, int((bb.ymin + bb.height) * h))

        return 1, [x1, y1, x2, y2]

    def __call__(self, image: np.ndarray) -> Tuple[int, List[int]]:
        return self.detect(image)
