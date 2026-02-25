import dlib
import cv2


class FaceDetector:
    """
    Production face detector wrapper
    """

    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def detect(self, image):
        """
        Returns list of face rectangles
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 1)
        return rects