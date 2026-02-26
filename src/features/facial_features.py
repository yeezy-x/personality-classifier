"""
facial_features.py

Unified Facial Feature Module
-----------------------------

Computes:
    - Eye ratio
    - Nose ratio
    - Face ratio
    - Cheekbone ratio
    - Double chin probability (CNN)

Designed for ContinuousPersonalityModel
"""

import cv2
import numpy as np
import tensorflow as tf


class FacialFeatureModule:

    def __init__(
        self,
        eye_threshold: float,
        nose_threshold: float,
        face_threshold: float,
        cheekbone_threshold: float,
        double_chin_model_path: str,
        double_chin_threshold: float,
    ):

        # Thresholds (still useful for internal classification if needed)
        self.eye_threshold = eye_threshold
        self.nose_threshold = nose_threshold
        self.face_threshold = face_threshold
        self.cheekbone_threshold = cheekbone_threshold
        self.double_chin_threshold = double_chin_threshold

        # Load CNN model (expects 224x224x3 unless trained differently)
        self.double_chin_model = tf.keras.models.load_model(double_chin_model_path)

    # -------------------------------------------------
    # Utility
    # -------------------------------------------------

    @staticmethod
    def _euclidean(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    # -------------------------------------------------
    # Geometric Ratios
    # -------------------------------------------------

    def _eye_ratio(self, shape):

        right_eye = shape[36:42]
        left_eye = shape[42:48]

        right_center = np.mean(right_eye, axis=0)
        left_center = np.mean(left_eye, axis=0)

        eye_distance = self._euclidean(left_center, right_center)
        face_width = self._euclidean(shape[0], shape[16])

        if face_width == 0:
            return 0.0

        return eye_distance / face_width

    def _nose_ratio(self, shape):

        nose_left = shape[31]
        nose_right = shape[35]

        nose_width = self._euclidean(nose_left, nose_right)
        face_width = self._euclidean(shape[0], shape[16])

        if face_width == 0:
            return 0.0

        return nose_width / face_width

    def _face_ratio(self, shape):

        jaw_left = shape[0]
        jaw_right = shape[16]
        chin = shape[8]

        eyebrow_points = shape[17:27]
        eyebrow_mid = np.mean(eyebrow_points, axis=0)

        face_width = self._euclidean(jaw_left, jaw_right)
        face_height = self._euclidean(eyebrow_mid, chin)

        if face_height == 0:
            return 0.0

        return face_width / face_height

    def _cheekbone_ratio(self, shape):

        cheek_left = shape[2]
        cheek_right = shape[14]

        jaw_left = shape[0]
        jaw_right = shape[16]

        cheek_width = self._euclidean(cheek_left, cheek_right)
        jaw_width = self._euclidean(jaw_left, jaw_right)

        if jaw_width == 0:
            return 0.0

        return cheek_width / jaw_width

    # -------------------------------------------------
    # Double Chin CNN
    # -------------------------------------------------

    def _double_chin_probability(self, image):

        # Resize to match model input
        img = cv2.resize(image, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        pred = self.double_chin_model.predict(img, verbose=0)[0]

        # Handle sigmoid vs softmax
        if len(pred) == 1:
            return float(pred[0])
        else:
            return float(pred[1])  # class index 1 = double chin

    # -------------------------------------------------
    # Public Extract Method
    # -------------------------------------------------

    def extract(self, image, shape):

        eye_ratio = self._eye_ratio(shape)
        nose_ratio = self._nose_ratio(shape)
        face_ratio = self._face_ratio(shape)
        cheekbone_ratio = self._cheekbone_ratio(shape)
        double_chin_prob = self._double_chin_probability(image)

        return {
            "eye_ratio": eye_ratio,
            "nose_ratio": nose_ratio,
            "face_ratio": face_ratio,
            "cheekbone_ratio": cheekbone_ratio,
            "double_chin_prob": double_chin_prob,
        }