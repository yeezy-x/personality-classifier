import pandas as pd
import numpy as np


class ContinuousPersonalityModel:
    """
    Continuous Weighted Linear Personality Model

    5 Inputs:
        eye_ratio
        nose_ratio
        face_ratio
        cheekbone_ratio
        double_chin_prob

    → 28 personality outputs
    """

    FEATURE_COLUMNS = [
        "eye_ratio",
        "nose_ratio",
        "face_ratio",
        "cheekbone_ratio",
        "double_chin_prob",
        "eye_face_interaction",
        "nose_chin_interaction",
        "face_cheek_interaction"
    ]

    INTERACTION_SCALE = 0.3

    def __init__(self, lookup_path):

        self.lookup = pd.read_csv(lookup_path).fillna(0)

        self.personality_dims = list(
            self.lookup.drop(columns=["Attributes"]).columns
        )

        self.weight_matrix = self._build_weight_matrix()

    # ---------------------------------------------------
    # Build Weight Matrix
    # ---------------------------------------------------

    def _build_weight_matrix(self):

        W = pd.DataFrame(
            0.0,
            index=self.personality_dims,
            columns=self.FEATURE_COLUMNS
        )

        trait_to_feature = {
            "Wide Set": "eye_ratio",
            "Close Set": "eye_ratio",
            "Large Nose": "nose_ratio",
            "Long Nose": "nose_ratio",
            "Broad Face": "face_ratio",
            "High Cheekbones": "cheekbone_ratio",
            "Double Chin": "double_chin_prob"
        }

        for _, row in self.lookup.iterrows():

            trait = row["Attributes"]

            if trait not in trait_to_feature:
                continue

            feature = trait_to_feature[trait]

            for dim in self.personality_dims:
                W.loc[dim, feature] += row[dim]

        # Add nonlinear interaction weights
        W["eye_face_interaction"] = self.INTERACTION_SCALE * W["eye_ratio"]
        W["nose_chin_interaction"] = self.INTERACTION_SCALE * W["nose_ratio"]
        W["face_cheek_interaction"] = self.INTERACTION_SCALE * W["face_ratio"]

        return W

    # ---------------------------------------------------
    # Feature Expansion
    # ---------------------------------------------------

    def _expand_features(self, fv):

        return np.array([
            fv["eye_ratio"],
            fv["nose_ratio"],
            fv["face_ratio"],
            fv["cheekbone_ratio"],
            fv["double_chin_prob"],
            fv["eye_ratio"] * fv["face_ratio"],
            fv["nose_ratio"] * fv["double_chin_prob"],
            fv["face_ratio"] * fv["cheekbone_ratio"]
        ], dtype=np.float32)

    # ---------------------------------------------------
    # Normalization (NEW)
    # ---------------------------------------------------

    def _normalize(self, X):
        """
        Z-score normalization to stabilize scale differences.
        """

        mean = np.mean(X)
        std = np.std(X) + 1e-8  # avoid division by zero

        return (X - mean) / std

    # ---------------------------------------------------
    # Predict
    # ---------------------------------------------------

    def predict(self, feature_vector):

        # Expand to 8D
        X = self._expand_features(feature_vector)

        # Normalize input vector
        X = self._normalize(X)

        W = self.weight_matrix.values

        scores = np.dot(W, X)

        result = dict(zip(self.personality_dims, scores))

        return dict(
            sorted(result.items(), key=lambda x: x[1], reverse=True)
        )