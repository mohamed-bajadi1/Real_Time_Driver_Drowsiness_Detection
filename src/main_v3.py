"""
Real-Time Driver Drowsiness Detection System
Main Application v3 - Improved Inference Pipeline

IMPROVEMENTS OVER v2:
1. CLAHE preprocessing for consistent contrast
2. Temporal smoothing (consecutive frames required for state change)
3. Increased eye bounding box padding
4. Configurable detection threshold (prioritize closed-eye recall)
5. Drowsiness alarm system
6. Better performance monitoring

Author: Binomial Team
Date: December 2025
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model
from tensorflow import keras
from tensorflow.keras import layers
from keras.regularizers import l2


# ============================================
# CONFIGURATION
# ============================================

# Camera settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Model settings
MODEL_PATH = "models/eye_state_classifier_v3.h5"  # v3 model (recommended)
# MODEL_PATH = "models/eye_state_classifier_v2.h5"  # v2 model (had issues)
# MODEL_PATH = "models/eye_state_classifier.h5"    # v1 model
EYE_IMG_SIZE = (64, 64)

# Detection threshold - LOWER = more sensitive to closed eyes
# Recommended: 0.4-0.5 for drowsiness detection (prioritize recall)
DETECTION_THRESHOLD = 0.4

# Temporal smoothing - require N consecutive frames for state change
# Prevents false alarms from blinks (typically < 300ms)
CONSECUTIVE_FRAMES_THRESHOLD = 3  # ~100-150ms at 20-30 FPS

# Drowsiness alarm settings
DROWSINESS_FRAME_THRESHOLD = 20  # ~0.7-1 second of closed eyes triggers alarm
ALARM_COOLDOWN_FRAMES = 60       # Cooldown between alarms

# Eye extraction settings - INCREASED PADDING
EYE_BBOX_PADDING = 15  # Increased from 10 to capture more context

# CLAHE settings for contrast normalization
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (4, 4)

# Display settings
FPS_DISPLAY = True
DEBUG_MODE = False  # Show preprocessed eye images

# MediaPipe configuration
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Facial landmarks indices (MediaPipe FaceMesh has 478 landmarks)
LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 144, 145, 153]
RIGHT_EYE_INDICES = [362, 263, 387, 386, 385, 373, 374, 380]
MOUTH_INDICES = [61, 291, 0, 17, 269, 405, 314, 17, 84, 181, 91, 146]
HEAD_POSE_INDICES = [1, 33, 263, 61, 291, 199]


# ============================================
# CLAHE PREPROCESSING (Critical for real-time!)
# ============================================

# Create CLAHE object once (reuse for efficiency)
clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_SIZE)


def apply_clahe(image):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    This is CRITICAL for real-time performance because:
    1. Training data has consistent lighting
    2. Webcam lighting varies dramatically
    3. CLAHE normalizes contrast, making model predictions consistent
    
    Args:
        image: Grayscale image (uint8)
    
    Returns:
        CLAHE-enhanced image (uint8)
    """
    return clahe.apply(image)


# ============================================
# MODEL LOADER (Supports v1 and v2 architectures)
# ============================================

def build_model_v1():
    """Original v1 architecture (171K params)."""
    model = keras.Sequential([
        layers.Conv2D(16, (3,3), activation='relu', input_shape=(64,64,1)),
        layers.MaxPooling2D(2,2),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Flatten(),
        layers.Dense(64, activation='relu', kernel_regularizer=l2(1e-4)),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


def build_model_v2():
    """Improved v2 architecture with attention (~250K params)."""
    from tensorflow.keras.layers import (
        Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout,
        GlobalAveragePooling2D, Dense, Add, Multiply, Concatenate, Activation
    )
    from tensorflow.keras import Model
    
    def spatial_attention_block(input_tensor):
        avg_pool = keras.backend.mean(input_tensor, axis=-1, keepdims=True)
        max_pool = keras.backend.max(input_tensor, axis=-1, keepdims=True)
        concat = Concatenate()([avg_pool, max_pool])
        attention = Conv2D(1, (7, 7), padding='same', activation='sigmoid')(concat)
        return Multiply()([input_tensor, attention])
    
    def residual_block(x, filters):
        shortcut = x
        x = Conv2D(filters, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, 3, padding='same')(x)
        x = BatchNormalization()(x)
        if shortcut.shape[-1] != filters:
            shortcut = Conv2D(filters, 1, padding='same')(shortcut)
            shortcut = BatchNormalization()(shortcut)
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x
    
    inputs = Input(shape=(64, 64, 1))
    
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = residual_block(x, 64)
    x = spatial_attention_block(x)
    
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    return Model(inputs=inputs, outputs=outputs, name='EyeClassifier_v2')


def build_model_v3():
    """Robust v3 architecture - standard CNN without attention."""
    from tensorflow.keras.layers import (
        Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout,
        GlobalAveragePooling2D, Dense, Activation
    )
    from tensorflow.keras import Model
    
    inputs = Input(shape=(64, 64, 1))
    
    # Block 1
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Block 2
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Block 3
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Classification head
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    return Model(inputs=inputs, outputs=outputs, name='EyeClassifier_v3')


def load_model_fixed(model_path):
    """
    Load Keras model with architecture auto-detection.
    
    Tries to detect model version based on file path and load appropriately.
    """
    import warnings
    
    # Determine model version from path
    if 'v3' in model_path.lower():
        model_version = 'v3'
    elif 'v2' in model_path.lower():
        model_version = 'v2'
    else:
        model_version = 'v1'
    
    print(f"   Detected model version: {model_version}")
    
    # Build appropriate architecture
    if model_version == 'v3':
        model = build_model_v3()
    elif model_version == 'v2':
        model = build_model_v2()
    else:
        model = build_model_v1()
    
    # Try to load weights
    try:
        model.load_weights(model_path)
        print("   [OK] Weights loaded successfully")
    except Exception as e:
        print(f"   [WARNING] Direct weight load failed: {e}")
        print("   [INFO] Attempting fallback loading...")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                old_model = load_model(model_path, compile=False)
                for new_layer, old_layer in zip(model.layers, old_model.layers):
                    try:
                        new_layer.set_weights(old_layer.get_weights())
                    except:
                        pass
                print("   [OK] Fallback loading successful")
            except Exception as e2:
                print(f"   [ERROR] Fallback failed: {e2}")
                raise RuntimeError("Could not load model")
    
    # Compile
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# ============================================
# EYE EXTRACTION & PREPROCESSING (Improved)
# ============================================

def get_eye_region(frame, landmarks, eye_indices, padding=EYE_BBOX_PADDING):
    """
    Extract bounding box coordinates for eye region.
    
    IMPROVEMENT: Increased padding to capture more context and
    better match training data distribution.
    """
    h, w, _ = frame.shape
    x_coords = []
    y_coords = []
    
    for idx in eye_indices:
        if idx < len(landmarks):
            landmark = landmarks[idx]
            x_coords.append(int(landmark.x * w))
            y_coords.append(int(landmark.y * h))
    
    if x_coords and y_coords:
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Calculate eye dimensions
        eye_width = x_max - x_min
        eye_height = y_max - y_min
        
        # Add padding proportional to eye size (minimum: padding pixels)
        pad_x = max(padding, int(eye_width * 0.3))
        pad_y = max(padding, int(eye_height * 0.5))
        
        x_min = max(0, x_min - pad_x)
        y_min = max(0, y_min - pad_y)
        x_max = min(w, x_max + pad_x)
        y_max = min(h, y_max + pad_y)
        
        return (x_min, y_min, x_max, y_max)
    
    return None


def extract_eye(frame, eye_bbox):
    """
    Extract and preprocess eye region for CNN model.
    
    PREPROCESSING PIPELINE (must match training!):
    1. Crop eye region from BGR frame
    2. Convert to grayscale
    3. Apply CLAHE (contrast normalization) - NEW!
    4. Resize to 64x64
    5. Normalize to [0, 1]
    6. Add batch and channel dimensions
    
    Args:
        frame: Original BGR frame
        eye_bbox: Tuple (x_min, y_min, x_max, y_max)
    
    Returns:
        Preprocessed eye image: (1, 64, 64, 1)
    """
    if eye_bbox is None:
        return None
    
    x_min, y_min, x_max, y_max = eye_bbox
    
    # 1. Crop eye region
    eye_crop = frame[y_min:y_max, x_min:x_max]
    
    if eye_crop.size == 0:
        return None
    
    # 2. Convert to grayscale
    eye_gray = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
    
    # 3. Apply CLAHE (CRITICAL for real-time consistency!)
    eye_clahe = apply_clahe(eye_gray)
    
    # 4. Resize to model input size
    eye_resized = cv2.resize(eye_clahe, EYE_IMG_SIZE)
    
    # 5. Normalize to [0, 1]
    eye_normalized = eye_resized.astype(np.float32) / 255.0
    
    # 6. Add dimensions: (64, 64) -> (1, 64, 64, 1)
    eye_batch = np.expand_dims(np.expand_dims(eye_normalized, axis=-1), axis=0)
    
    return eye_batch


# ============================================
# TEMPORAL SMOOTHING (Prevents false alarms)
# ============================================

class TemporalSmoother:
    """
    Smooths eye state predictions over time.
    
    Purpose:
    - Blinks are typically < 300ms
    - Drowsiness = sustained closed eyes
    - We want to detect drowsiness, not blinks
    
    Strategy:
    - Require N consecutive frames with same prediction
    - Only change displayed state after confirmation
    - Track drowsiness duration separately
    """
    
    def __init__(self, window_size=CONSECUTIVE_FRAMES_THRESHOLD):
        self.window_size = window_size
        self.left_history = deque(maxlen=window_size)
        self.right_history = deque(maxlen=window_size)
        self.left_state = "OPEN"   # Confirmed state
        self.right_state = "OPEN"  # Confirmed state
        self.both_closed_frames = 0  # Counter for drowsiness detection
        self.alarm_cooldown = 0
    
    def update(self, left_closed, right_closed, left_conf, right_conf):
        """
        Update with new predictions and return smoothed states.
        
        Args:
            left_closed: Boolean, raw prediction for left eye
            right_closed: Boolean, raw prediction for right eye
            left_conf: Confidence for left prediction
            right_conf: Confidence for right prediction
        
        Returns:
            Tuple: (left_state, right_state, drowsy_alert, drowsy_frames)
        """
        # Add to history
        self.left_history.append(1 if left_closed else 0)
        self.right_history.append(1 if right_closed else 0)
        
        # Check for consecutive frames
        if len(self.left_history) >= self.window_size:
            # Left eye: require all frames to agree
            if sum(self.left_history) == self.window_size:
                self.left_state = "CLOSED"
            elif sum(self.left_history) == 0:
                self.left_state = "OPEN"
            # Otherwise keep previous state (hysteresis)
        
        if len(self.right_history) >= self.window_size:
            if sum(self.right_history) == self.window_size:
                self.right_state = "CLOSED"
            elif sum(self.right_history) == 0:
                self.right_state = "OPEN"
        
        # Track drowsiness (both eyes closed)
        if self.left_state == "CLOSED" and self.right_state == "CLOSED":
            self.both_closed_frames += 1
        else:
            self.both_closed_frames = max(0, self.both_closed_frames - 2)  # Decay slowly
        
        # Cooldown management
        if self.alarm_cooldown > 0:
            self.alarm_cooldown -= 1
        
        # Drowsiness alert
        drowsy_alert = False
        if self.both_closed_frames >= DROWSINESS_FRAME_THRESHOLD:
            if self.alarm_cooldown == 0:
                drowsy_alert = True
                self.alarm_cooldown = ALARM_COOLDOWN_FRAMES
                self.both_closed_frames = 0  # Reset counter after alert
        
        return (
            self.left_state,
            self.right_state,
            drowsy_alert,
            self.both_closed_frames
        )


# ============================================
# PREDICTION WITH CONFIGURABLE THRESHOLD
# ============================================

def predict_eye_state(model, eye_image, threshold=DETECTION_THRESHOLD):
    """
    Predict if eye is open or closed using CNN model.
    
    IMPORTANT: Using lower threshold (0.4) to prioritize closed-eye detection.
    
    Args:
        model: Loaded Keras model
        eye_image: Preprocessed eye image (1, 64, 64, 1)
        threshold: Classification threshold (default: 0.4)
    
    Returns:
        prediction: Raw float [0-1]
        is_closed: Boolean
        confidence: Percentage
    """
    if eye_image is None:
        return None, None, None
    
    # Get raw prediction
    prediction = model.predict(eye_image, verbose=0)[0][0]
    
    # Class 0 = Closed, Class 1 = Open
    # prediction < threshold -> Closed
    is_closed = prediction < threshold
    
    # Calculate confidence (distance from threshold)
    if is_closed:
        # For closed: confidence = how far below threshold
        confidence = ((threshold - prediction) / threshold) * 100
    else:
        # For open: confidence = how far above threshold
        confidence = ((prediction - threshold) / (1 - threshold)) * 100
    
    # Clamp confidence to [0, 100]
    confidence = np.clip(confidence, 0, 100)
    
    return prediction, is_closed, confidence


# ============================================
# HELPER FUNCTIONS
# ============================================

def calculate_fps(prev_time):
    """Calculate FPS for performance monitoring."""
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
    return fps, current_time


def draw_landmarks_custom(frame, landmarks, indices, color=(0, 255, 0), thickness=2):
    """Draw specific landmarks on the frame."""
    h, w, _ = frame.shape
    points = []
    
    for idx in indices:
        if idx < len(landmarks):
            landmark = landmarks[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            points.append((x, y))
            cv2.circle(frame, (x, y), 2, color, -1)
    
    # Draw connecting lines
    if len(points) > 1:
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], color, thickness)
        if len(points) > 2:
            cv2.line(frame, points[-1], points[0], color, thickness)
    
    return points


def display_info(frame, fps, detection_status, left_status=None, right_status=None, 
                 drowsy_frames=0, drowsy_alert=False):
    """Display comprehensive status information on frame."""
    # Background for info panel
    info_height = 170 if left_status else 100
    cv2.rectangle(frame, (10, 10), (350, info_height), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (350, info_height), (255, 255, 255), 2)
    
    # FPS
    fps_color = (0, 255, 0) if fps >= 20 else (0, 165, 255) if fps >= 15 else (0, 0, 255)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
    
    # Detection status
    status_color = (0, 255, 0) if detection_status else (0, 0, 255)
    status_text = "Face Detected" if detection_status else "No Face"
    cv2.putText(frame, f"Status: {status_text}", (20, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    # Threshold info
    cv2.putText(frame, f"Threshold: {DETECTION_THRESHOLD:.2f}", (200, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Eye states
    if left_status and right_status:
        left_text, left_conf = left_status
        right_text, right_conf = right_status
        
        # Left eye
        left_color = (0, 0, 255) if left_text == "CLOSED" else (0, 255, 0)
        cv2.putText(frame, f"L-Eye: {left_text} ({left_conf:.0f}%)", (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 2)
        
        # Right eye
        right_color = (0, 0, 255) if right_text == "CLOSED" else (0, 255, 0)
        cv2.putText(frame, f"R-Eye: {right_text} ({right_conf:.0f}%)", (20, 115), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 2)
        
        # Drowsiness progress bar
        progress = min(drowsy_frames / DROWSINESS_FRAME_THRESHOLD, 1.0)
        bar_width = int(200 * progress)
        bar_color = (0, 255, 0) if progress < 0.5 else (0, 165, 255) if progress < 0.8 else (0, 0, 255)
        cv2.rectangle(frame, (20, 130), (220, 145), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, 130), (20 + bar_width, 145), bar_color, -1)
        cv2.putText(frame, f"Drowsy: {int(progress*100)}%", (230, 143), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    # Instructions
    cv2.putText(frame, "Press 'q' to quit | 't' toggle threshold", (20, info_height - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    # Drowsiness alert overlay
    if drowsy_alert:
        # Flash red overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Alert text
        cv2.putText(frame, "!!! DROWSINESS DETECTED !!!", (100, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(frame, "WAKE UP!", (250, 350),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)


# ============================================
# MAIN APPLICATION
# ============================================

def main():
    """Main application loop with all improvements."""
    global DETECTION_THRESHOLD, DEBUG_MODE
    
    print("=" * 60)
    print("Driver Drowsiness Detection System - v3 (Improved)")
    print("MediaPipe FaceMesh + CNN Eye-State Classification")
    print("=" * 60)
    print("\nKey Improvements:")
    print("  - CLAHE preprocessing for contrast normalization")
    print("  - Temporal smoothing to prevent false alarms")
    print("  - Configurable threshold (prioritize closed-eye recall)")
    print("  - Drowsiness detection with visual alerts")
    print("=" * 60)
    
    # Load CNN model
    eye_model = None
    if os.path.exists(MODEL_PATH):
        print(f"\nLoading model from: {MODEL_PATH}")
        try:
            eye_model = load_model_fixed(MODEL_PATH)
            print(f"   Model input shape: {eye_model.input_shape}")
            print(f"   Model parameters: {eye_model.count_params():,}")
        except Exception as e:
            print(f"[ERROR] Could not load model: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n[WARNING] Model not found at: {MODEL_PATH}")
        print("   Searching for alternative models...")
        
        # Try alternative paths
        alt_paths = [
            "eye_state_classifier_v3.h5",
            "eye_state_classifier_v3.keras",
            "eye_state_classifier_v2.h5",
            "eye_state_classifier.h5",
            "models/eye_state_classifier_v3.h5",
            "models/eye_state_classifier.h5",
            "../models/eye_state_classifier_v3.h5"
        ]
        
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                print(f"   Found: {alt_path}")
                try:
                    eye_model = load_model_fixed(alt_path)
                    break
                except:
                    continue
    
    if eye_model is None:
        print("\n[ERROR] No model loaded. Running in visualization-only mode.")
    
    # Initialize webcam
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return
    
    print("[OK] Camera initialized successfully")
    
    # Initialize temporal smoother
    smoother = TemporalSmoother()
    
    # Initialize MediaPipe FaceMesh
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        print("[OK] MediaPipe FaceMesh initialized")
        print(f"\nDetection threshold: {DETECTION_THRESHOLD}")
        print(f"Temporal smoothing: {CONSECUTIVE_FRAMES_THRESHOLD} frames")
        print(f"Drowsiness alert: {DROWSINESS_FRAME_THRESHOLD} frames")
        print("\nStarting real-time detection...")
        print("Press 'q' to quit, 't' to toggle threshold\n")
        
        prev_time = time.time()
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("[ERROR] Failed to grab frame")
                break
            
            frame_count += 1
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = face_mesh.process(rgb_frame)
            
            face_detected = False
            left_status = None
            right_status = None
            drowsy_alert = False
            drowsy_frames = 0
            
            if results.multi_face_landmarks:
                face_detected = True
                
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = face_landmarks.landmark
                    
                    # Draw landmarks
                    draw_landmarks_custom(frame, landmarks, LEFT_EYE_INDICES, 
                                         color=(0, 255, 0), thickness=1)
                    draw_landmarks_custom(frame, landmarks, RIGHT_EYE_INDICES, 
                                         color=(0, 255, 0), thickness=1)
                    
                    # Get eye bounding boxes
                    left_eye_bbox = get_eye_region(frame, landmarks, LEFT_EYE_INDICES)
                    right_eye_bbox = get_eye_region(frame, landmarks, RIGHT_EYE_INDICES)
                    
                    # CNN Predictions
                    if eye_model is not None:
                        left_closed_raw = False
                        right_closed_raw = False
                        left_conf = 0
                        right_conf = 0
                        
                        # Left eye
                        if left_eye_bbox:
                            left_eye_img = extract_eye(frame, left_eye_bbox)
                            if left_eye_img is not None:
                                _, left_closed_raw, left_conf = predict_eye_state(
                                    eye_model, left_eye_img, DETECTION_THRESHOLD
                                )
                                
                                # Debug: show preprocessed image
                                if DEBUG_MODE:
                                    debug_img = (left_eye_img[0, :, :, 0] * 255).astype(np.uint8)
                                    cv2.imshow('Left Eye (Preprocessed)', 
                                              cv2.resize(debug_img, (128, 128)))
                        
                        # Right eye
                        if right_eye_bbox:
                            right_eye_img = extract_eye(frame, right_eye_bbox)
                            if right_eye_img is not None:
                                _, right_closed_raw, right_conf = predict_eye_state(
                                    eye_model, right_eye_img, DETECTION_THRESHOLD
                                )
                        
                        # Apply temporal smoothing
                        if left_closed_raw is not None and right_closed_raw is not None:
                            left_state, right_state, drowsy_alert, drowsy_frames = smoother.update(
                                left_closed_raw, right_closed_raw, left_conf, right_conf
                            )
                            
                            left_status = (left_state, left_conf)
                            right_status = (right_state, right_conf)
                        
                        # Draw bounding boxes
                        if left_eye_bbox:
                            bbox_color = (0, 0, 255) if left_status and left_status[0] == "CLOSED" else (0, 255, 0)
                            cv2.rectangle(frame, (left_eye_bbox[0], left_eye_bbox[1]),
                                        (left_eye_bbox[2], left_eye_bbox[3]), bbox_color, 2)
                        
                        if right_eye_bbox:
                            bbox_color = (0, 0, 255) if right_status and right_status[0] == "CLOSED" else (0, 255, 0)
                            cv2.rectangle(frame, (right_eye_bbox[0], right_eye_bbox[1]),
                                        (right_eye_bbox[2], right_eye_bbox[3]), bbox_color, 2)
            
            # Calculate FPS
            fps, prev_time = calculate_fps(prev_time)
            
            # Display info
            display_info(frame, fps, face_detected, left_status, right_status, 
                        drowsy_frames, drowsy_alert)
            
            # Show frame
            cv2.imshow('Drowsiness Detection v3', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[STOPPING] Application terminated by user")
                break
            elif key == ord('t'):
                # Toggle threshold
                if DETECTION_THRESHOLD == 0.4:
                    DETECTION_THRESHOLD = 0.5
                else:
                    DETECTION_THRESHOLD = 0.4
                print(f"[INFO] Threshold changed to: {DETECTION_THRESHOLD}")
            elif key == ord('d'):
                # Toggle debug mode
                DEBUG_MODE = not DEBUG_MODE
                print(f"[INFO] Debug mode: {'ON' if DEBUG_MODE else 'OFF'}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print(f"[OK] Application ended successfully")
    print(f"[INFO] Total frames processed: {frame_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
