import dlib
import cv2
from imutils import face_utils


class LandmarkExtractor:
    """
    Extracts 68 facial landmarks
    """

    def __init__(self, predictor_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def get_landmarks(self, image):
        """
        Returns:
            shape (np.array 68x2) or None
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 1)

        # Only accept single face images
        if len(rects) != 1:
            return None

        shape = self.predictor(gray, rects[0])
        shape = face_utils.shape_to_np(shape)

        return shape
    
    def visualize(self, image, shape):
        """
        Draw landmarks (debug only)
        """
        import cv2

        output = image.copy()

        for (x, y) in shape:
            cv2.circle(output, (x, y), 2, (0, 0, 255), -1)

        return output