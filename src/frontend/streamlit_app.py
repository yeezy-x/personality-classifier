import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import time

from src.pipeline.feature_pipeline import FeaturePipeline
from src.inference.personality_linear_model import ContinuousPersonalityModel
from src.feature_extraction.landmark_extractor import LandmarkExtractor


# -----------------------------
# CONFIG
# -----------------------------

CONFIG = {
    "predictor_path": "models/landmark_model/shape_predictor_68_face_landmarks.dat",
    "double_chin_model_path": "models/double_chin_model.h5",
    "lookup_path": "data/raw/lookup.csv",
    "eye_threshold": 0.43,
    "nose_threshold": 0.19,
    "face_threshold": 0.85,
    "cheek_threshold": 0.95,
    "double_chin_threshold": 0.61
}

# -----------------------------
# INIT
# -----------------------------

@st.cache_resource
def load_pipeline():
    return FeaturePipeline(**CONFIG)

@st.cache_resource
def load_personality_model():
    return ContinuousPersonalityModel(CONFIG["lookup_path"])

pipeline = load_pipeline()
personality_model = load_personality_model()

if "step" not in st.session_state:
    st.session_state.step = 1

st.title("🧠 AI Facial Personality Analyzer")

# -----------------------------
# STEP 1 — Upload
# -----------------------------

if st.session_state.step == 1:

    uploaded_file = st.file_uploader("Upload an Image", type=["jpg","jpeg","png"])

    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        st.session_state.image_path = tfile.name

        image = cv2.imread(st.session_state.image_path)
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if st.button("Proceed to Face Detection"):
            st.session_state.step = 2
            st.rerun()

# -----------------------------
# STEP 2 — Bounding Box
# -----------------------------

elif st.session_state.step == 2:

    image = cv2.imread(st.session_state.image_path)
    extractor = LandmarkExtractor(CONFIG["predictor_path"])
    shape = extractor.get_landmarks(image)

    if shape is None:
        st.error("Face not detected or multiple faces.")
        st.stop()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = extractor.detector(gray, 1)

    bbox_img = image.copy()
    for rect in rects:
        cv2.rectangle(
            bbox_img,
            (rect.left(), rect.top()),
            (rect.right(), rect.bottom()),
            (0,255,0),3
        )

    st.subheader("Face Detection")
    st.image(cv2.cvtColor(bbox_img, cv2.COLOR_BGR2RGB))

    if st.button("Proceed to Landmark Detection"):
        st.session_state.step = 3
        st.rerun()

# -----------------------------
# STEP 3 — Landmarks
# -----------------------------

elif st.session_state.step == 3:

    image = cv2.imread(st.session_state.image_path)
    extractor = LandmarkExtractor(CONFIG["predictor_path"])
    shape = extractor.get_landmarks(image)

    landmark_img = extractor.visualize(image, shape)

    st.subheader("Landmark Detection")
    st.image(cv2.cvtColor(landmark_img, cv2.COLOR_BGR2RGB))

    if st.button("Proceed to Feature Analysis"):
        st.session_state.step = 4
        st.rerun()

# -----------------------------
# STEP 4 — Ratio Analysis
# -----------------------------

elif st.session_state.step == 4:

    features = pipeline.extract_features(st.session_state.image_path)
    st.session_state.ratios = features["ratios"]

    st.subheader("Facial Ratio Analysis")

    fig, ax = plt.subplots()
    ax.bar(features["ratios"].keys(), features["ratios"].values())
    plt.xticks(rotation=45)
    st.pyplot(fig)

    if st.button("Detect Personality"):
        st.session_state.step = 5
        st.rerun()

# -----------------------------
# STEP 5 — Loader
# -----------------------------

elif st.session_state.step == 5:

    st.subheader("Detecting Personality...")
    progress = st.progress(0)

    for i in range(100):
        time.sleep(0.02)
        progress.progress(i+1)

    personality_scores = personality_model.predict(
        st.session_state.ratios
    )

    st.session_state.personality = personality_scores
    st.session_state.step = 6
    st.rerun()

# -----------------------------
# STEP 6 — Final Dashboard
# -----------------------------

elif st.session_state.step == 6:

    st.success("Personality Analysis Complete!")

    tabs = st.tabs(["Top Traits","Radar Chart","Full Scores"])

    # Tab 1
    with tabs[0]:
        top_traits = list(st.session_state.personality.items())[:5]
        for trait, score in top_traits:
            st.write(f"**{trait}** : {round(score,3)}")

    # Tab 2
    with tabs[1]:
        traits = list(st.session_state.personality.keys())[:8]
        values = list(st.session_state.personality.values())[:8]

        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
        angles = np.linspace(0, 2*np.pi, len(traits), endpoint=False)
        values += values[:1]
        angles = np.concatenate((angles,[angles[0]]))

        ax.plot(angles, values)
        ax.fill(angles, values, alpha=0.3)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(traits)
        st.pyplot(fig)

    # Tab 3
    with tabs[2]:
        st.json(st.session_state.personality)

    if st.button("Start Over"):
        st.session_state.step = 1
        st.rerun()