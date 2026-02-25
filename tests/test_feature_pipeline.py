from src.pipeline.feature_pipeline import FeaturePipeline

def run_test():

    pipeline = FeaturePipeline(
        predictor_path="models/landmark_model/shape_predictor_68_face_landmarks.dat",
        double_chin_model_path="models/double_chin_model.h5",
        eye_threshold=0.43,
        nose_threshold=0.33,
        face_threshold=0.87,
        cheek_threshold=0.95,
    )

    image_path = "sample_images/test.jpg"

    features = pipeline.extract_features(image_path)

    print("Extracted Features:")
    print(features)


if __name__ == "__main__":
    run_test()