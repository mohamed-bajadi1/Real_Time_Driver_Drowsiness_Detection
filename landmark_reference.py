"""
MediaPipe FaceMesh Landmark Reference Guide
============================================

This file documents the specific landmark indices used in our drowsiness detection system.
MediaPipe FaceMesh provides 478 landmarks (468 face + 10 iris).

Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
"""

# ============================================
# EYE LANDMARKS (for Eye-State Classification)
# ============================================

# Left Eye - 8 key points forming the eye contour
LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 144, 145, 153]
"""
Left Eye Landmarks:
- 33:  Left eye inner corner
- 133: Left eye outer corner  
- 160: Top-left of eye
- 159: Top-center of eye
- 158: Top-right of eye
- 144: Bottom-left of eye
- 145: Bottom-center of eye
- 153: Bottom-right of eye

Usage: These points form a closed contour around the left eye.
We use this to:
1. Draw the eye region visualization
2. Calculate bounding box for CNN input
3. Crop the eye image (64x64) for classification
"""

# Right Eye - 8 key points forming the eye contour
RIGHT_EYE_INDICES = [362, 263, 387, 386, 385, 373, 374, 380]
"""
Right Eye Landmarks:
- 362: Right eye inner corner
- 263: Right eye outer corner
- 387: Top-right of eye
- 386: Top-center of eye
- 385: Top-left of eye
- 373: Bottom-right of eye
- 374: Bottom-center of eye
- 380: Bottom-left of eye

Usage: Same as left eye - for visualization, bounding box, and CNN input.
"""

# Alternative: Eye Aspect Ratio (EAR) landmarks (not currently used)
# We use CNN instead of EAR for better accuracy
LEFT_EYE_EAR = [33, 160, 144, 133, 159, 145]  # 6-point EAR calculation
RIGHT_EYE_EAR = [362, 387, 373, 263, 386, 374]


# ============================================
# MOUTH LANDMARKS (for Yawn Detection - MAR)
# ============================================

MOUTH_INDICES = [61, 291, 0, 17, 269, 405, 314, 17, 84, 181, 91, 146]
"""
Mouth Landmarks - Outer lip contour:
- 61:  Left corner of mouth
- 291: Right corner of mouth
- 0:   Top center of upper lip
- 17:  Bottom center of lower lip
- Others: Points along lip contour

Usage: Calculate Mouth Aspect Ratio (MAR)
Formula: MAR = (vertical distance) / (horizontal distance)
- Yawn detected when MAR > threshold (typically 0.6-0.7)
- Must persist for several frames to avoid false positives
"""

# Alternative: More precise mouth landmarks for MAR
MOUTH_MAR_VERTICAL = [13, 14]   # Top and bottom lip center
MOUTH_MAR_HORIZONTAL = [61, 291]  # Left and right corners


# ============================================
# HEAD POSE LANDMARKS (for Head-Nod Detection)
# ============================================

HEAD_POSE_INDICES = [1, 33, 263, 61, 291, 199]
"""
Head Pose Reference Points:
- 1:   Nose tip (center)
- 33:  Left eye inner corner
- 263: Right eye inner corner
- 61:  Left mouth corner
- 291: Right mouth corner
- 199: Chin center

Usage: 3D Head Pose Estimation
- Calculate rotation matrix (pitch, yaw, roll)
- Pitch angle: Head nodding up/down
- Yaw angle: Head turning left/right
- Roll angle: Head tilting

Head nod detection:
- Track pitch angle over time
- Nod detected when pitch > threshold (typically 15-20 degrees)
"""

# Camera calibration parameters (for 3D pose estimation)
# These are approximate values for a standard webcam
CAMERA_MATRIX_DEFAULT = {
    'focal_length': 1.0,  # Normalized
    'center': (0.5, 0.5)  # Image center (normalized)
}


# ============================================
# IRIS LANDMARKS (optional - for advanced eye tracking)
# ============================================

# MediaPipe provides refined iris landmarks (indices 468-477)
# Not currently used, but available for future enhancements
LEFT_IRIS = [468, 469, 470, 471, 472]   # Left iris center + contour
RIGHT_IRIS = [473, 474, 475, 476, 477]  # Right iris center + contour


# ============================================
# UTILITY FUNCTIONS FOR LANDMARKS
# ============================================

def get_landmark_coords(landmarks, indices, frame_shape):
    """
    Extract (x, y) coordinates for specific landmark indices
    
    Args:
        landmarks: MediaPipe FaceMesh landmarks
        indices: List of landmark indices to extract
        frame_shape: (height, width, channels) of the frame
        
    Returns:
        List of (x, y) tuples in pixel coordinates
    """
    h, w = frame_shape[:2]
    coords = []
    for idx in indices:
        if idx < len(landmarks):
            lm = landmarks[idx]
            x = int(lm.x * w)
            y = int(lm.y * h)
            coords.append((x, y))
    return coords


def calculate_ear(eye_landmarks):
    """
    Calculate Eye Aspect Ratio (classical method)
    
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    
    Where:
    - p1, p4 = horizontal eye landmarks (corners)
    - p2, p3, p5, p6 = vertical eye landmarks (top/bottom)
    
    Returns:
        float: EAR value (typically 0.2-0.3 for open eyes, <0.2 for closed)
    """
    import numpy as np
    
    # Horizontal distance
    horizontal = np.linalg.norm(
        np.array(eye_landmarks[0]) - np.array(eye_landmarks[3])
    )
    
    # Vertical distances
    vertical1 = np.linalg.norm(
        np.array(eye_landmarks[1]) - np.array(eye_landmarks[5])
    )
    vertical2 = np.linalg.norm(
        np.array(eye_landmarks[2]) - np.array(eye_landmarks[4])
    )
    
    # EAR formula
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear


def calculate_mar(mouth_landmarks):
    """
    Calculate Mouth Aspect Ratio (for yawn detection)
    
    MAR = (||p2-p8|| + ||p3-p7|| + ||p4-p6||) / (3 * ||p1-p5||)
    
    Where:
    - p1, p5 = horizontal mouth landmarks (corners)
    - p2, p3, p4, p6, p7, p8 = vertical mouth landmarks
    
    Returns:
        float: MAR value (typically <0.5 for closed, >0.6 for yawning)
    """
    import numpy as np
    
    # Horizontal distance
    horizontal = np.linalg.norm(
        np.array(mouth_landmarks[0]) - np.array(mouth_landmarks[4])
    )
    
    # Vertical distances (multiple measurements for better accuracy)
    vertical1 = np.linalg.norm(
        np.array(mouth_landmarks[1]) - np.array(mouth_landmarks[7])
    )
    vertical2 = np.linalg.norm(
        np.array(mouth_landmarks[2]) - np.array(mouth_landmarks[6])
    )
    vertical3 = np.linalg.norm(
        np.array(mouth_landmarks[3]) - np.array(mouth_landmarks[5])
    )
    
    # MAR formula
    mar = (vertical1 + vertical2 + vertical3) / (3.0 * horizontal)
    return mar


# ============================================
# CONFIGURATION & THRESHOLDS
# ============================================

# Thresholds for drowsiness detection (to be tuned experimentally)
THRESHOLDS = {
    # Eye-state detection
    'EAR_THRESHOLD': 0.2,              # EAR below this = closed eye
    'EYE_CLOSED_FRAMES': 60,           # ~2 seconds at 30 FPS
    
    # Yawn detection
    'MAR_THRESHOLD': 0.6,              # MAR above this = yawn
    'YAWN_FRAMES': 20,                 # ~0.67 seconds at 30 FPS
    
    # Head nod detection
    'HEAD_NOD_THRESHOLD': 20,          # Pitch angle in degrees
    'HEAD_NOD_FRAMES': 30,             # ~1 second at 30 FPS
    
    # Integrated drowsiness score
    'DROWSINESS_SCORE_THRESHOLD': 3.0, # Combined score threshold
}

# Scoring weights for integrated system
SCORING_WEIGHTS = {
    'eye_closed': 1.0,    # Base weight for closed eyes
    'yawn': 0.5,          # Lower weight (less critical)
    'head_nod': 0.8,      # Medium weight
}


# ============================================
# VISUAL CONFIGURATION
# ============================================

COLORS = {
    'eye_landmarks': (0, 255, 0),      # Green
    'mouth_landmarks': (255, 0, 0),    # Blue
    'head_landmarks': (0, 0, 255),     # Red
    'eye_bbox': (0, 255, 255),         # Yellow (Cyan)
    'alarm': (0, 0, 255),              # Red
    'warning': (0, 165, 255),          # Orange
    'safe': (0, 255, 0),               # Green
}


if __name__ == "__main__":
    """
    Display landmark information when run directly
    """
    print("=" * 70)
    print("MediaPipe FaceMesh Landmark Reference")
    print("=" * 70)
    print()
    
    print("📍 Eye Landmarks (CNN Classification)")
    print(f"   Left Eye:  {LEFT_EYE_INDICES}")
    print(f"   Right Eye: {RIGHT_EYE_INDICES}")
    print()
    
    print("📍 Mouth Landmarks (MAR Calculation)")
    print(f"   Mouth Contour: {MOUTH_INDICES}")
    print()
    
    print("📍 Head Pose Landmarks (3D Orientation)")
    print(f"   Reference Points: {HEAD_POSE_INDICES}")
    print()
    
    print("🎯 Detection Thresholds")
    print("-" * 70)
    for key, value in THRESHOLDS.items():
        print(f"   {key}: {value}")
    print()
    
    print("⚖️ Scoring Weights")
    print("-" * 70)
    for key, value in SCORING_WEIGHTS.items():
        print(f"   {key}: {value}")
    print()
    
    print("=" * 70)
    print("For visual reference, see:")
    print("https://github.com/google/mediapipe/blob/master/docs/solutions/face_mesh.md")
    print("=" * 70)
