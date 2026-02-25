import cv2
import numpy as np

from src.feature_extraction.landmark_extractor import LandmarkExtractor
from src.feature_extraction.geometric_features import (
    extract_eye_ratio,
    extract_nose_ratio,
    extract_face_shape_ratio,
    extract_cheekbone_ratio,
    classify_face_shape,
)

import tensorflow as tf


class FeaturePipeline:
    """
    Full Hybrid Feature Extraction Pipeline
    ----------------------------------------
    Extracts:
        - Eye Spacing
        - Nose Width
        - Face Shape
        - Cheekbone Prominence
        - Double Chin (CNN)
    """

    def __init__(
        self,
        predictor_path,
        double_chin_model_path,
        eye_threshold,
        nose_threshold,
        face_threshold,
        cheek_threshold,
    ):

        # Landmark model
        self.landmark_extractor = LandmarkExtractor(predictor_path)

        # CNN model
        self.double_chin_model = tf.keras.models.load_model(double_chin_model_path)

        # Thresholds
        self.eye_threshold = eye_threshold
        self.nose_threshold = nose_threshold
        self.face_threshold = face_threshold
        self.cheek_threshold = cheek_threshold

    # ---------------------------------------------------
    # Double Chin CNN Prediction
    # ---------------------------------------------------

    def _predict_double_chin(self, image):
        """
        Runs CNN on full face image.
        Assumes model trained on resized 178x218.
        """

        resized = cv2.resize(image, (178, 218))
        normalized = resized / 255.0
        input_tensor = np.expand_dims(normalized, axis=0)

        prediction = self.double_chin_model.predict(input_tensor, verbose=0)

        # Binary softmax output
        class_index = np.argmax(prediction)

        return int(class_index)  # 1 = double chin, 0 = no

    # ---------------------------------------------------
    # Main Feature Extraction
    # ---------------------------------------------------

    def extract_features(self, image_path):
        """
        Returns structured dictionary of all 5 traits
        """

        image = cv2.imread(image_path)

        if image is None:
            return {"error": "Image could not be loaded"}

        shape = self.landmark_extractor.get_landmarks(image)

        if shape is None:
            return {"error": "Face not detected or multiple faces"}

        # ---------------------------
        # Geometric Ratios
        # ---------------------------

        eye_ratio = extract_eye_ratio(shape)
        nose_ratio = extract_nose_ratio(shape)
        face_ratio = extract_face_shape_ratio(shape)
        cheek_ratio = extract_cheekbone_ratio(shape)

        # ---------------------------
        # Classifications
        # ---------------------------

        wide_set = 1 if eye_ratio > self.eye_threshold else 0
        big_nose = 1 if nose_ratio > self.nose_threshold else 0
        broad_face = 1 if face_ratio > self.face_threshold else 0
        high_cheekbone = 1 if cheek_ratio > self.cheek_threshold else 0

        double_chin = self._predict_double_chin(image)

        # ---------------------------
        # Structured Output
        # ---------------------------

        return {
            "eye_spacing_ratio": float(eye_ratio),
            "nose_width_ratio": float(nose_ratio),
            "face_shape_ratio": float(face_ratio),
            "cheekbone_ratio": float(cheek_ratio),

            "wide_set_eyes": wide_set,
            "big_nose": big_nose,
            "broad_face": broad_face,
            "high_cheekbones": high_cheekbone,
            "double_chin": double_chin,
        }