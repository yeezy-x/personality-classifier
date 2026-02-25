import numpy as np


# ==========================================================
# Utility
# ==========================================================

def _euclidean(p1, p2):
    """Compute Euclidean distance between two points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))


# ==========================================================
# Core Geometric Ratios
# ==========================================================

def extract_eye_ratio(shape):
    """
    Interocular distance normalized by face width.
    """

    right_eye = shape[36:42]
    left_eye = shape[42:48]

    right_center = np.mean(right_eye, axis=0)
    left_center = np.mean(left_eye, axis=0)

    eye_distance = _euclidean(left_center, right_center)
    face_width = _euclidean(shape[0], shape[16])

    if face_width == 0:
        return None

    return eye_distance / face_width


def extract_nose_ratio(shape):
    """
    Nose width normalized by face width.
    """

    nose_left = shape[31]
    nose_right = shape[35]

    nose_width = _euclidean(nose_left, nose_right)
    face_width = _euclidean(shape[0], shape[16])

    if face_width == 0:
        return None

    return nose_width / face_width


def extract_mouth_ratio(shape):
    """
    Mouth width normalized by face width.
    """

    mouth_left = shape[48]
    mouth_right = shape[54]

    mouth_width = _euclidean(mouth_left, mouth_right)
    face_width = _euclidean(shape[0], shape[16])

    if face_width == 0:
        return None

    return mouth_width / face_width


def extract_jaw_ratio(shape):
    """
    Face width normalized by face height.
    """

    jaw_left = shape[0]
    jaw_right = shape[16]
    chin = shape[8]
    face_top = shape[27]

    face_width = _euclidean(jaw_left, jaw_right)
    face_height = _euclidean(face_top, chin)

    if face_height == 0:
        return None

    return face_width / face_height

def extract_face_shape_ratio(shape):
    """
    Face width / face height

    Width  : Jaw extremes (0,16)
    Height : Eyebrow midpoint (17–26 mean) to Chin (8)

    Expected realistic range:
        ~0.70 – 0.95
    """

    # Width
    left_jaw = shape[0]
    right_jaw = shape[16]
    face_width = _euclidean(left_jaw, right_jaw)

    # Height (stable top reference)
    eyebrow_points = shape[17:27]
    eyebrow_mid = np.mean(eyebrow_points, axis=0)

    chin = shape[8]
    face_height = _euclidean(eyebrow_mid, chin)

    if face_height == 0:
        return None

    return face_width / face_height

def extract_cheekbone_ratio(shape):
    """
    Cheekbone width / Jaw width

    Cheekbone width : distance between points 2 and 14
    Jaw width       : distance between points 0 and 16

    Expected range:
        0.80 – 1.05

    >1  → very prominent cheekbones
    <0.85 → softer / rounder midface
    """

    import numpy as np

    def euclidean(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    cheek_left = shape[2]
    cheek_right = shape[14]

    jaw_left = shape[0]
    jaw_right = shape[16]

    cheek_width = euclidean(cheek_left, cheek_right)
    jaw_width = euclidean(jaw_left, jaw_right)

    if jaw_width == 0:
        return None

    return cheek_width / jaw_width

# ==========================================================
# Classifiers
# ==========================================================

def classify_eye_spacing(shape, threshold):
    """
    Returns:
        1 → Wide set
        0 → Close set
        None → Invalid
    """
    ratio = extract_eye_ratio(shape)

    if ratio is None:
        return None

    return 1 if ratio > threshold else 0


def classify_big_nose(shape, threshold):
    """
    Returns:
        1 → Big nose
        0 → Normal
        None → Invalid
    """
    ratio = extract_nose_ratio(shape)

    if ratio is None:
        return None

    return 1 if ratio > threshold else 0

def classify_face_shape(shape, threshold):
    """
    Returns:
        1 → Broad/Round
        0 → Long/Slim
        None → Invalid
    """

    ratio = extract_face_shape_ratio(shape)

    if ratio is None:
        return None

    return 1 if ratio > threshold else 0


def classify_cheekbone(shape, threshold):
    ratio = extract_cheekbone_ratio(shape)

    if ratio is None:
        return None

    return 1 if ratio > threshold else 0




# ==========================================================
# Unified Feature Extractor
# ==========================================================

def extract_geometric_features(
    shape,
    eye_threshold,
    nose_threshold,
    face_threshold,
    cheekbone_threshold
):
    """
    Returns structured geometric feature dictionary.
    """

    eye_ratio = extract_eye_ratio(shape)
    nose_ratio = extract_nose_ratio(shape)
    mouth_ratio = extract_mouth_ratio(shape)
    jaw_ratio = extract_jaw_ratio(shape)
    face_ratio = extract_face_shape_ratio(shape)


    features = {
        "eye_spacing_ratio": eye_ratio,
        "nose_width_ratio": nose_ratio,
        "mouth_width_ratio": mouth_ratio,
        "jaw_width_ratio": jaw_ratio,
        "face_shape_ratio": face_ratio,
        "is_wide_set": classify_eye_spacing(shape, eye_threshold),
        "is_big_nose": classify_big_nose(shape, nose_threshold),
        "is_broad_face": classify_face_shape(shape, face_threshold),
        "is_prominent_cheekbones": classify_cheekbone(shape, cheekbone_threshold)
    }

    return features