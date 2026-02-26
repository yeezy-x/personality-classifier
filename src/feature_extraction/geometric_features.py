"""
facial_features.py

Production-grade feature extraction module.

Includes:
- Geometric traits (Explainable)
- CNN-based double chin inference
- Structured scalable output
"""

import numpy as np
import cv2
import tensorflow as tf


# ==========================================================
# Utility
# ==========================================================

def _euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


# ==========================================================
# Geometric Feature Extractor
# ==========================================================

class GeometricFeatureExtractor:
    """
    Extracts explainable geometric facial traits
    from 68-point landmark shape.
    """

    def __init__(
        self,
        eye_threshold: float,
        nose_threshold: float,
        face_threshold: float,
        cheekbone_threshold: float
    ):
        self.eye_threshold = eye_threshold
        self.nose_threshold = nose_threshold
        self.face_threshold = face_threshold
        self.cheekbone_threshold = cheekbone_threshold

    # -------------------------
    # Core Ratios
    # -------------------------

    def eye_ratio(self, shape):
        right_eye = shape[36:42]
        left_eye = shape[42:48]

        right_center = np.mean(right_eye, axis=0)
        left_center = np.mean(left_eye, axis=0)

        eye_distance = _euclidean(left_center, right_center)
        face_width = _euclidean(shape[0], shape[16])

        return None if face_width == 0 else eye_distance / face_width

    def nose_ratio(self, shape):
        nose_left = shape[31]
        nose_right = shape[35]

        nose_width = _euclidean(nose_left, nose_right)
        face_width = _euclidean(shape[0], shape[16])

        return None if face_width == 0 else nose_width / face_width

    def face_shape_ratio(self, shape):
        left_jaw = shape[0]
        right_jaw = shape[16]
        face_width = _euclidean(left_jaw, right_jaw)

        eyebrow_mid = np.mean(shape[17:27], axis=0)
        chin = shape[8]

        face_height = _euclidean(eyebrow_mid, chin)

        return None if face_height == 0 else face_width / face_height

    def cheekbone_ratio(self, shape):
        cheek_left = shape[2]
        cheek_right = shape[14]

        cheek_width = _euclidean(cheek_left, cheek_right)
        jaw_width = _euclidean(shape[0], shape[16])

        return None if jaw_width == 0 else cheek_width / jaw_width

    # -------------------------
    # Classification
    # -------------------------

    def classify(self, shape):
        eye_r = self.eye_ratio(shape)
        nose_r = self.nose_ratio(shape)
        face_r = self.face_shape_ratio(shape)
        cheek_r = self.cheekbone_ratio(shape)

        return {
            "eye_ratio": eye_r,
            "nose_ratio": nose_r,
            "face_ratio": face_r,
            "cheekbone_ratio": cheek_r,

            "is_wide_set": None if eye_r is None else eye_r > self.eye_threshold,
            "is_big_nose": None if nose_r is None else nose_r > self.nose_threshold,
            "is_broad_face": None if face_r is None else face_r > self.face_threshold,
            "is_high_cheekbones": None if cheek_r is None else cheek_r > self.cheekbone_threshold
        }


# ==========================================================
# Double Chin CNN Feature
# ==========================================================

class DoubleChinModel:
    """
    CNN-based Double Chin classifier
    """

    def __init__(self, model_path: str, threshold: float):
        self.model = tf.keras.models.load_model(model_path)
        self.threshold = threshold
        self.img_size = (178, 218)

    def _prepare(self, image):
        img = cv2.resize(image, self.img_size)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    def predict(self, image):
        """
        Returns:
            {
                "double_chin_prob": float,
                "has_double_chin": bool,
                "confidence": float
            }
        """

        img = self._prepare(image)
        prob = float(self.model.predict(img, verbose=0)[0][0])

        has_double = prob > self.threshold
        confidence = prob if has_double else (1 - prob)

        return {
            "double_chin_prob": prob,
            "has_double_chin": has_double,
            "confidence": confidence
        }


# ==========================================================
# Unified Facial Feature Module
# ==========================================================

class FacialFeatureModule:
    """
    Combines geometric + CNN traits.
    """

    def __init__(
        self,
        eye_threshold,
        nose_threshold,
        face_threshold,
        cheekbone_threshold,
        double_chin_model_path,
        double_chin_threshold
    ):

        self.geo = GeometricFeatureExtractor(
            eye_threshold,
            nose_threshold,
            face_threshold,
            cheekbone_threshold
        )

        self.double_chin = DoubleChinModel(
            double_chin_model_path,
            double_chin_threshold
        )

    def extract(self, image, shape):
        """
        image  : original image (BGR)
        shape  : 68-point numpy array

        Returns structured feature dictionary.
        """

        geo_features = self.geo.classify(shape)
        dc_features = self.double_chin.predict(image)

        return {
            **geo_features,
            **dc_features
        }