"""
Real-Time Driver Drowsiness Detection System
Main Application with MediaPipe FaceMesh + CNN Eye-State Classification
Phase 2: CNN Integration
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from tensorflow.keras.models import load_model
import os

# ============================================
# CONFIGURATION
# ============================================
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS_DISPLAY = True

# Model configuration
MODEL_PATH = "C:\\Users\\dell\\Desktop\\IAII\\Deep Learning\\DL_Project\\models\\eye_state_classifier.h5"
EYE_IMG_SIZE = (64, 64)  # Model expects 64x64 grayscale input
USE_MODEL = True  # Set to False to run without model (landmark visualization only)

# MediaPipe configuration
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Facial landmarks indices (MediaPipe FaceMesh has 478 landmarks)
# Eyes landmarks
LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 144, 145, 153]  # Left eye contour
RIGHT_EYE_INDICES = [362, 263, 387, 386, 385, 373, 374, 380]  # Right eye contour

# Mouth landmarks
MOUTH_INDICES = [61, 291, 0, 17, 269, 405, 314, 17, 84, 181, 91, 146]  # Mouth contour

# Head pose landmarks (nose tip, chin, forehead, etc.)
HEAD_POSE_INDICES = [1, 33, 263, 61, 291, 199]  # Key points for head orientation


# ============================================
# CUSTOM MODEL LOADER (Fix for TF 2.15+ compatibility)
# ============================================
def load_model_fixed(model_path):
    """
    Load Keras model with compatibility fix for TensorFlow 2.15+

    Since the old H5 format has compatibility issues, we recreate the model
    architecture and load the weights.
    """
    from tensorflow import keras
    from tensorflow.keras import layers
    from keras.regularizers import l2

    # Recreate the model architecture from training notebook
    # This MUST match train-eye-model.ipynb exactly!
    model = keras.Sequential([
        # Block 1
        layers.Conv2D(16, (3,3), activation='relu', input_shape=(64,64,1)),
        layers.MaxPooling2D(2,2),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        # Block 2
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Block 3
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        # Dense layers
        layers.Flatten(),
        layers.Dense(64, activation='relu', kernel_regularizer=l2(1e-4)),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    # Load weights from H5 file
    try:
        model.load_weights(model_path)
    except Exception as e:
        print(f"   [WARNING] Could not load weights directly: {e}")
        print(f"   [INFO] Attempting fallback loading method...")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                old_model = load_model(model_path, compile=False)
                # Copy weights layer by layer
                for new_layer, old_layer in zip(model.layers, old_model.layers):
                    try:
                        new_layer.set_weights(old_layer.get_weights())
                    except:
                        pass
            except:
                raise RuntimeError("Could not load model. Please retrain with current TensorFlow version.")

    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# ============================================
# EYE EXTRACTION & PREPROCESSING
# ============================================

def extract_eye(frame, eye_bbox):
    """
    Extract and preprocess eye region for CNN model
    
    Args:
        frame: Original BGR frame
        eye_bbox: Tuple (x_min, y_min, x_max, y_max)
    
    Returns:
        Preprocessed eye image ready for model prediction
        Shape: (1, 64, 64, 1) - batch_size, height, width, channels
    """
    if eye_bbox is None:
        return None
    
    x_min, y_min, x_max, y_max = eye_bbox
    
    # Crop eye region from frame
    eye_crop = frame[y_min:y_max, x_min:x_max]
    
    # Check if crop is valid
    if eye_crop.size == 0:
        return None
    
    # Convert to grayscale (model was trained on grayscale)
    eye_gray = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
    
    # Resize to 64x64 (model input size)
    eye_resized = cv2.resize(eye_gray, EYE_IMG_SIZE)
    
    # Normalize to [0, 1] (same as training preprocessing)
    eye_normalized = eye_resized / 255.0
    
    # Add channel dimension: (64, 64) -> (64, 64, 1)
    eye_normalized = np.expand_dims(eye_normalized, axis=-1)
    
    # Add batch dimension: (64, 64, 1) -> (1, 64, 64, 1)
    eye_batch = np.expand_dims(eye_normalized, axis=0)
    
    return eye_batch


def predict_eye_state(model, eye_image):
    """
    Predict if eye is open or closed using CNN model
    
    Args:
        model: Loaded Keras model
        eye_image: Preprocessed eye image (1, 64, 64, 1)
    
    Returns:
        prediction: Float between 0-1 (0=Closed, 1=Open)
        is_closed: Boolean indicating if eye is closed
        confidence: Prediction confidence percentage
    """
    if eye_image is None:
        return None, None, None
    
    # Get prediction
    prediction = model.predict(eye_image, verbose=0)[0][0]
    
    # Class 0 = Closed, Class 1 = Open (from training data)
    is_closed = prediction < 0.5
    
    # Calculate confidence
    confidence = (1 - prediction) * 100 if is_closed else prediction * 100
    
    return prediction, is_closed, confidence


# ============================================
# HELPER FUNCTIONS
# ============================================

def calculate_fps(prev_time):
    """Calculate FPS for performance monitoring"""
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    return fps, current_time


def draw_landmarks_custom(frame, landmarks, indices, color=(0, 255, 0), thickness=2):
    """Draw specific landmarks on the frame"""
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
        # Close the loop
        if len(points) > 2:
            cv2.line(frame, points[-1], points[0], color, thickness)
    
    return points


def get_eye_region(frame, landmarks, eye_indices):
    """Extract bounding box coordinates for eye region"""
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
        
        # Add padding
        padding = 10
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        return (x_min, y_min, x_max, y_max)
    
    return None


def display_info(frame, fps, detection_status, left_status=None, right_status=None):
    """Display FPS, detection status, and eye states on frame"""
    # Background for text
    info_height = 140 if left_status else 100
    cv2.rectangle(frame, (10, 10), (320, info_height), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (320, info_height), (255, 255, 255), 2)
    
    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Detection status
    status_color = (0, 255, 0) if detection_status else (0, 0, 255)
    status_text = "Face Detected" if detection_status else "No Face"
    cv2.putText(frame, f"Status: {status_text}", (20, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    # Eye states (if available)
    if left_status and right_status:
        left_text, left_conf = left_status
        right_text, right_conf = right_status
        
        # Left eye
        left_color = (0, 0, 255) if left_text == "CLOSED" else (0, 255, 0)
        cv2.putText(frame, f"L-Eye: {left_text} ({left_conf:.0f}%)", (20, 95), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 2)
        
        # Right eye
        right_color = (0, 0, 255) if right_text == "CLOSED" else (0, 255, 0)
        cv2.putText(frame, f"R-Eye: {right_text} ({right_conf:.0f}%)", (20, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 2)
    
    # Instructions
    cv2.putText(frame, "Press 'q' to quit", (20, info_height - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


# ============================================
# MAIN APPLICATION
# ============================================

def main():
    """Main application loop"""
    print("=" * 60)
    print("Driver Drowsiness Detection System - Phase 2")
    print("MediaPipe FaceMesh + CNN Eye-State Classification")
    print("=" * 60)
    
    # Load CNN model (if enabled)
    eye_model = None
    if USE_MODEL:
        if os.path.exists(MODEL_PATH):
            print(f"\nLoading model from: {MODEL_PATH}")
            try:
                eye_model = load_model_fixed(MODEL_PATH)
                print("[OK] Model loaded successfully")
                print(f"   Model input shape: {eye_model.input_shape}")
                print(f"   Model output shape: {eye_model.output_shape}")
            except Exception as e:
                print(f"[ERROR] Error loading model: {e}")
                print("   Running in landmark visualization mode only")
                import traceback
                traceback.print_exc()
        else:
            print(f"\n[WARNING] Model not found at: {MODEL_PATH}")
            print("   Running in landmark visualization mode only")
            print("   Please ensure eye_state_classifier.h5 is in the models/ directory")
    
    # Initialize webcam
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    if not cap.isOpened():
        print("[ERROR] Error: Cannot open camera")
        return

    print("[OK] Camera initialized successfully")

    # Initialize MediaPipe FaceMesh
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        print("[OK] MediaPipe FaceMesh initialized")
        print("\nStarting real-time detection...")
        print("Press 'q' to quit\n")
        
        prev_time = time.time()
        frame_count = 0
        
        while True:
            ret, frame = cap.read()

            if not ret:
                print("[ERROR] Failed to grab frame")
                break
            
            frame_count += 1
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with MediaPipe
            results = face_mesh.process(rgb_frame)
            
            face_detected = False
            left_status = None
            right_status = None
            
            # Draw landmarks if face is detected
            if results.multi_face_landmarks:
                face_detected = True
                
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = face_landmarks.landmark
                    
                    # Draw left eye (GREEN)
                    left_eye_points = draw_landmarks_custom(
                        frame, landmarks, LEFT_EYE_INDICES, 
                        color=(0, 255, 0), thickness=2
                    )
                    
                    # Draw right eye (GREEN)
                    right_eye_points = draw_landmarks_custom(
                        frame, landmarks, RIGHT_EYE_INDICES, 
                        color=(0, 255, 0), thickness=2
                    )
                    
                    # Draw mouth (BLUE)
                    mouth_points = draw_landmarks_custom(
                        frame, landmarks, MOUTH_INDICES, 
                        color=(255, 0, 0), thickness=2
                    )
                    
                    # Draw head pose reference points (RED)
                    head_points = draw_landmarks_custom(
                        frame, landmarks, HEAD_POSE_INDICES, 
                        color=(0, 0, 255), thickness=1
                    )
                    
                    # Get eye bounding boxes
                    left_eye_bbox = get_eye_region(frame, landmarks, LEFT_EYE_INDICES)
                    right_eye_bbox = get_eye_region(frame, landmarks, RIGHT_EYE_INDICES)
                    
                    # CNN Prediction (if model is loaded)
                    if eye_model is not None:
                        # Extract and predict left eye
                        if left_eye_bbox:
                            left_eye_img = extract_eye(frame, left_eye_bbox)
                            if left_eye_img is not None:
                                _, left_closed, left_conf = predict_eye_state(eye_model, left_eye_img)
                                left_status = ("CLOSED" if left_closed else "OPEN", left_conf)
                                
                                # Draw bbox with color based on state
                                bbox_color = (0, 0, 255) if left_closed else (0, 255, 0)
                                cv2.rectangle(frame, (left_eye_bbox[0], left_eye_bbox[1]),
                                            (left_eye_bbox[2], left_eye_bbox[3]), 
                                            bbox_color, 2)
                                cv2.putText(frame, f"L: {left_status[0]}", 
                                          (left_eye_bbox[0], left_eye_bbox[1] - 5),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, bbox_color, 1)
                        
                        # Extract and predict right eye
                        if right_eye_bbox:
                            right_eye_img = extract_eye(frame, right_eye_bbox)
                            if right_eye_img is not None:
                                _, right_closed, right_conf = predict_eye_state(eye_model, right_eye_img)
                                right_status = ("CLOSED" if right_closed else "OPEN", right_conf)
                                
                                # Draw bbox with color based on state
                                bbox_color = (0, 0, 255) if right_closed else (0, 255, 0)
                                cv2.rectangle(frame, (right_eye_bbox[0], right_eye_bbox[1]),
                                            (right_eye_bbox[2], right_eye_bbox[3]), 
                                            bbox_color, 2)
                                cv2.putText(frame, f"R: {right_status[0]}", 
                                          (right_eye_bbox[0], right_eye_bbox[1] - 5),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, bbox_color, 1)
                    else:
                        # No model - just draw yellow boxes
                        if left_eye_bbox:
                            cv2.rectangle(frame, (left_eye_bbox[0], left_eye_bbox[1]),
                                        (left_eye_bbox[2], left_eye_bbox[3]), 
                                        (0, 255, 255), 2)
                            cv2.putText(frame, "L-Eye", (left_eye_bbox[0], left_eye_bbox[1] - 5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                        
                        if right_eye_bbox:
                            cv2.rectangle(frame, (right_eye_bbox[0], right_eye_bbox[1]),
                                        (right_eye_bbox[2], right_eye_bbox[3]), 
                                        (0, 255, 255), 2)
                            cv2.putText(frame, "R-Eye", (right_eye_bbox[0], right_eye_bbox[1] - 5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # Calculate and display FPS
            if FPS_DISPLAY:
                fps, prev_time = calculate_fps(prev_time)
                display_info(frame, fps, face_detected, left_status, right_status)
            
            # Display the frame
            window_title = 'Drowsiness Detection - Phase 2 (CNN Integrated)'
            cv2.imshow(window_title, frame)
            
            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[STOPPING] Stopping application...")
                break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print(f"[OK] Application ended successfully")
    print(f"[INFO] Total frames processed: {frame_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
