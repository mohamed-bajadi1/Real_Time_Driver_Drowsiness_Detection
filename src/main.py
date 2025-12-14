"""
Real-Time Driver Drowsiness Detection System
Main Application with MediaPipe FaceMesh Integration
"""

import cv2
import mediapipe as mp
import numpy as np
import time

# ============================================
# CONFIGURATION
# ============================================
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS_DISPLAY = True

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


def display_info(frame, fps, detection_status):
    """Display FPS and system information on frame"""
    # Background for text
    cv2.rectangle(frame, (10, 10), (300, 100), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (300, 100), (255, 255, 255), 2)
    
    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Detection status
    status_color = (0, 255, 0) if detection_status else (0, 0, 255)
    status_text = "Face Detected" if detection_status else "No Face"
    cv2.putText(frame, f"Status: {status_text}", (20, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    # Instructions
    cv2.putText(frame, "Press 'q' to quit", (20, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


# ============================================
# MAIN APPLICATION
# ============================================

def main():
    """Main application loop"""
    print("=" * 60)
    print("Driver Drowsiness Detection System - Phase 1")
    print("MediaPipe FaceMesh Integration")
    print("=" * 60)
    
    # Initialize webcam
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    if not cap.isOpened():
        print("❌ Error: Cannot open camera")
        return
    
    print("✅ Camera initialized successfully")
    
    # Initialize MediaPipe FaceMesh
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        print("✅ MediaPipe FaceMesh initialized")
        print("\nStarting real-time detection...")
        print("Press 'q' to quit\n")
        
        prev_time = time.time()
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Failed to grab frame")
                break
            
            frame_count += 1
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with MediaPipe
            results = face_mesh.process(rgb_frame)
            
            face_detected = False
            
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
                    
                    # Get and draw eye bounding boxes
                    left_eye_bbox = get_eye_region(frame, landmarks, LEFT_EYE_INDICES)
                    right_eye_bbox = get_eye_region(frame, landmarks, RIGHT_EYE_INDICES)
                    
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
                display_info(frame, fps, face_detected)
            
            # Display the frame
            cv2.imshow('Driver Drowsiness Detection - FaceMesh', frame)
            
            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n🛑 Stopping application...")
                break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print(f"✅ Application ended successfully")
    print(f"📊 Total frames processed: {frame_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
