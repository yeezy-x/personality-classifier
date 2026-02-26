from src.pipeline.feature_pipeline import FeaturePipeline
from src.inference.personality_linear_model import ContinuousPersonalityModel


CONFIG = {
    "predictor_path": "models/landmark_model/shape_predictor_68_face_landmarks.dat",
    "double_chin_model_path": "models/double_chin_model.h5",
    "lookup_path": "data/raw/lookup.csv",

    "eye_threshold": 0.43,
    "nose_threshold": 0.19,
    "face_threshold": 0.85 ,#1.05 
    "cheek_threshold": 0.95,
    "double_chin_threshold": 0.61
}


def run_personality_pipeline(image_path):

    pipeline = FeaturePipeline(
        predictor_path=CONFIG["predictor_path"],
        double_chin_model_path=CONFIG["double_chin_model_path"],
        eye_threshold=CONFIG["eye_threshold"],
        nose_threshold=CONFIG["nose_threshold"],
        face_threshold=CONFIG["face_threshold"],
        cheek_threshold=CONFIG["cheek_threshold"],
        double_chin_threshold=CONFIG["double_chin_threshold"]
    )

    features = pipeline.extract_features(image_path)

    if "error" in features:
        return features

    model = ContinuousPersonalityModel(CONFIG["lookup_path"])

    personality_scores = model.predict(features["ratios"])

    return {
        "ratios": features["ratios"],
        "personality_scores": personality_scores
    }



if __name__ == "__main__":

    output = run_personality_pipeline("/home/yeezy/100xDevs/personality/Personality-Classifier/sample_images/face.jpg")

    if "error" in output:
        print(output["error"])
    else:
        print("\nRatios:")
        for k, v in output["ratios"].items():
            print(f"{k}: {round(v, 4)}")

        print("\nTop Personality Traits:")
        for k, v in list(output["personality_scores"].items())[:5]:
            print(f"{k}: {round(v, 3)}")