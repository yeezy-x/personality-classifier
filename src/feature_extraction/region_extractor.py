import os
import cv2
import dlib
import imutils
import numpy as np
from imutils import face_utils
from collections import OrderedDict


# ------------------------------------------------
# Facial landmark index mapping
# ------------------------------------------------
FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("inner_mouth", (60, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])


class RegionExtractor:
    def __init__(self, predictor_path):
        """
        Initialize dlib face detector and landmark predictor.
        """
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def extract_region(self, image, region_name, resize_width=250, padding=40):
        """
        Extract a specific facial region from an image.

        Parameters:
            image (numpy array): Input image
            region_name (str): Name of region (jaw, nose, mouth, etc.)
            resize_width (int): Resize output width
            padding (int): Extra pixels below region (useful for jaw)

        Returns:
            Cropped region image or None
        """

        if region_name not in FACIAL_LANDMARKS_IDXS:
            raise ValueError(f"Invalid region name: {region_name}")

        image = imutils.resize(image, width=600)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        rects = self.detector(gray, 1)

        if len(rects) != 1:
            return None

        for rect in rects:
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            (start, end) = FACIAL_LANDMARKS_IDXS[region_name]
            (x, y, w, h) = cv2.boundingRect(np.array([shape[start:end]]))

            y_start = max(0, y)
            y_end = min(image.shape[0], y + h + padding)
            x_start = max(0, x)
            x_end = min(image.shape[1], x + w)

            roi = image[y_start:y_end, x_start:x_end]

            try:
                roi = imutils.resize(roi, width=resize_width)
                return roi
            except:
                return None

        return None

    def process_folder(self, input_folder, output_folder, region_name):
        """
        Extract region for all images in folder.
        """
        os.makedirs(output_folder, exist_ok=True)

        processed = 0
        skipped = 0

        for file in os.listdir(input_folder):

            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, file)

            image = cv2.imread(input_path)

            if image is None:
                skipped += 1
                continue

            roi = self.extract_region(image, region_name)

            if roi is not None:
                cv2.imwrite(output_path, roi)
                processed += 1
            else:
                skipped += 1

        print(f"{region_name} extraction -> Processed: {processed}, Skipped: {skipped}")