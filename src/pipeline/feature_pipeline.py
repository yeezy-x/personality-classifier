"""
feature_pipeline.py

Hybrid Feature Extraction Pipeline
-----------------------------------

Extracts 5 continuous facial metrics used by:
ContinuousPersonalityModel

Outputs:
{
    "ratios": {
        eye_ratio,
        nose_ratio,
        face_ratio,
        cheekbone_ratio,
        double_chin_prob
    }
}
"""

import cv2
from typing import Dict, Any

from src.feature_extraction.landmark_extractor import LandmarkExtractor
from src.features.facial_features import FacialFeatureModule


class FeaturePipeline:
    """
    Hybrid Feature Pipeline

    Designed specifically for ContinuousPersonalityModel.
    Returns only continuous features required by ANN layer.
    """

    def __init__(
        self,
        predictor_path: str,
        double_chin_model_path: str,
        eye_threshold: float,
        nose_threshold: float,
        face_threshold: float,
        cheek_threshold: float,
        double_chin_threshold: float,
    ):

        # Landmark extractor (dlib 68-point)
        self.landmark_extractor = LandmarkExtractor(predictor_path)

        # Unified facial feature module
        self.feature_module = FacialFeatureModule(
            eye_threshold=eye_threshold,
            nose_threshold=nose_threshold,
            face_threshold=face_threshold,
            cheekbone_threshold=cheek_threshold,
            double_chin_model_path=double_chin_model_path,
            double_chin_threshold=double_chin_threshold,
        )

    # ---------------------------------------------------
    # Main Extraction
    # ---------------------------------------------------

    def extract_features(self, image_path: str) -> Dict[str, Any]:
        """
        Extract continuous facial ratios only.

        Returns:
        {
            "ratios": {
                eye_ratio: float,
                nose_ratio: float,
                face_ratio: float,
                cheekbone_ratio: float,
                double_chin_prob: float
            }
        }
        """

        # ---------------------------------------
        # Load image
        # ---------------------------------------
        image = cv2.imread(image_path)

        if image is None:
            return {"error": "Image could not be loaded"}

        # ---------------------------------------
        # Extract facial landmarks
        # ---------------------------------------
        shape = self.landmark_extractor.get_landmarks(image)

        if shape is None:
            return {"error": "Face not detected or multiple faces"}

        # ---------------------------------------
        # Extract geometric + CNN features
        # ---------------------------------------
        result = self.feature_module.extract(image=image, shape=shape)

        if result is None:
            return {"error": "Feature extraction failed"}

        # ---------------------------------------
        # Only return continuous ratios
        # (Exactly what ANN model expects)
        # ---------------------------------------

        ratios = {
            "eye_ratio": float(result["eye_ratio"]),
            "nose_ratio": float(result["nose_ratio"]),
            "face_ratio": float(result["face_ratio"]),
            "cheekbone_ratio": float(result["cheekbone_ratio"]),
            "double_chin_prob": float(result["double_chin_prob"]),
        }

        return {
            "ratios": ratios
        }