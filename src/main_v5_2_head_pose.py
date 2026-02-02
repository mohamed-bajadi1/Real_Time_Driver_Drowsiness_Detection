"""
Real-Time Driver Drowsiness Detection System
Main Application v5.2 - WITH HEAD POSE ESTIMATION

CHANGES FROM v5.1:
1. Integrated HeadPoseEstimator for 3D head orientation tracking
2. Head pose passed to eye detectors for false alarm reduction
3. Head drooping detection as additional drowsiness signal
4. Visualization of head pose angles and state

DETECTION LOGIC:
- CNN provides primary signal (appearance-based)
- EAR provides geometric validation (landmark-based)
- Head pose discounts fusion when looking down briefly
- Head drooping (pitch < -30° for 2s) triggers alarm independently

ALARM TRIGGERS:
- Both eyes CLOSED for 20 frames → Alarm
- Eyes CLOSED + recent yawn → Faster alarm (10 frames)
- Excessive yawning (≥5 yawns + ≥6/min) → Alarm
- Head drooping (pitch < -30° for 60 frames) → Alarm [NEW]
- Severe head droop (pitch < -45° for 30 frames) → Immediate alarm [NEW]

Author: Binomial Team (Sprint 5)
Date: December 2025
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import os
import pygame

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# Import enhanced detector
from enhanced_eye_detector import EnhancedEyeStateDetector, DualEyeDetector, EyeState, calculate_ear

# Import head pose estimator
from head_pose_estimator import (
    HeadPoseEstimator,
    HeadPoseResult,
    draw_head_pose_axis,
    draw_head_pose_info
)

# ============================================
# CONFIGURATION
# ============================================

# Audio settings
ALARM_VOLUME = 0.7
ENABLE_AUDIO = True

# Path resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
ALARM_SOUND_PATH = os.path.join(PROJECT_ROOT, "data", "sounds", "deep.wav")

# Camera settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Model settings
MODEL_PATH = "../models/eye_state_classifier_v4.h5"
TFLITE_PATH = "../models/eye_state_classifier_v4.tflite"
USE_TFLITE = True

EYE_IMG_SIZE = (64, 64)

# Eye detection - NOTE: CNN outputs P(open), threshold means < this = closed
# The enhanced detector handles thresholding internally with fusion
DETECTION_THRESHOLD = 0.4  # Kept for reference / CNN confidence display

# Drowsiness settings
DROWSINESS_FRAME_THRESHOLD = 20  # Frames of closed eyes for alarm
ALARM_COOLDOWN_FRAMES = 60

# Yawn-based fatigue detection
YAWN_MIN_COUNT_FOR_ALARM = 5
YAWN_OBSERVATION_WINDOW = 180
YAWN_FATIGUE_THRESHOLD = 4
YAWN_DANGER_THRESHOLD = 6
YAWN_RECENT_WINDOW = 120

# Eye extraction
EYE_BBOX_PADDING = 15

# CLAHE settings
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (4, 4)

# Skip-frame mode
SKIP_FRAME_MODE = False
CNN_EVERY_N_FRAMES = 3

# ============================================
# HEAD POSE CONFIGURATION (NEW)
# ============================================
HEAD_POSE_SMOOTHING = 0.4           # EMA smoothing factor
HEAD_LOOKING_DOWN_THRESHOLD = -15.0  # Degrees - triggers discount
HEAD_SEVERE_DROOP_THRESHOLD = -30.0  # Degrees - drowsiness indicator
HEAD_LOOKING_AWAY_THRESHOLD = 30.0   # Degrees - side looking detection
HEAD_DROOP_ALARM_FRAMES = 60         # 2 seconds at 30 FPS
HEAD_SEVERE_DROOP_FRAMES = 30        # 1 second for immediate alarm
SHOW_HEAD_POSE_VISUALIZATION = True  # Draw 3D axis on face

# ============================================
# CONSOLE LOGGING CONFIGURATION (NEW FOR DEMO)
# ============================================
ENABLE_CONSOLE_LOGGING = True        # Toggle detailed console output
CONSOLE_LOG_EVERY_N_FRAMES = 30      # Log every N frames (30 = once per second at 30fps)
CONSOLE_LOG_STATE_CHANGES = True     # Log when head state changes
CONSOLE_LOG_ALARMS = True            # Always log alarm triggers

# ============================================
# YAWN DETECTION CONFIGURATION
# ============================================
YAWN_MAR_THRESHOLD = 0.55
YAWN_TEMPORAL_WINDOW = 15
YAWN_THRESHOLD_FRAMES = 8
YAWN_MIN_DURATION = 20
YAWN_COOLDOWN_FRAMES = 30

# MediaPipe configuration
mp_face_mesh = mp.solutions.face_mesh

# Facial landmarks indices
LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 144, 145, 153]
RIGHT_EYE_INDICES = [362, 263, 387, 386, 385, 373, 374, 380]

# Mouth landmarks for MAR
MOUTH_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
               409, 270, 269, 267, 0, 37, 39, 40, 185]

# Create CLAHE object
clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_SIZE)


# ============================================
# YAWN DETECTOR CLASS (unchanged from v5.1)
# ============================================

class YawnDetectorMAR:
    """MAR-based yawn detection."""
    
    UPPER_LIP_CENTER = 13
    LOWER_LIP_CENTER = 14
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291
    UPPER_LIP_LEFT = 82
    LOWER_LIP_LEFT = 87
    UPPER_LIP_RIGHT = 312
    LOWER_LIP_RIGHT = 317
    
    def __init__(self, mar_threshold=0.55, temporal_window=15,
                 yawn_threshold=8, min_yawn_duration=20, cooldown_frames=30):
        self.mar_threshold = mar_threshold
        self.temporal_window = temporal_window
        self.yawn_threshold = yawn_threshold
        self.min_yawn_duration = min_yawn_duration
        self.cooldown_frames = cooldown_frames
        
        self.yawn_window = deque(maxlen=temporal_window)
        self.mar_history = deque(maxlen=5)
        self.current_yawn_frames = 0
        self.yawn_count = 0
        self.cooldown_counter = 0
        self.in_yawn = False
        self.peak_mar = 0.0
        self.session_start = time.time()
        self.yawn_timestamps = deque(maxlen=30)
    
    def calculate_mar(self, landmarks, h, w):
        try:
            vertical_pairs = [
                (self.UPPER_LIP_CENTER, self.LOWER_LIP_CENTER),
                (self.UPPER_LIP_LEFT, self.LOWER_LIP_LEFT),
                (self.UPPER_LIP_RIGHT, self.LOWER_LIP_RIGHT),
            ]
            
            vertical_distances = []
            for upper_idx, lower_idx in vertical_pairs:
                upper = landmarks[upper_idx]
                lower = landmarks[lower_idx]
                dist = np.sqrt((upper.x - lower.x)**2 + (upper.y - lower.y)**2)
                vertical_distances.append(dist)
            
            avg_vertical = np.mean(vertical_distances)
            
            left = landmarks[self.MOUTH_LEFT]
            right = landmarks[self.MOUTH_RIGHT]
            horizontal = np.sqrt((left.x - right.x)**2 + (left.y - right.y)**2)
            
            if horizontal < 1e-6:
                return 0.0, False
            
            return avg_vertical / horizontal, True
        except:
            return 0.0, False
    
    def update(self, landmarks, h, w):
        timestamp = time.time()
        
        raw_mar, success = self.calculate_mar(landmarks, h, w)
        if not success:
            return False, 0.0, self._make_info(0.0, False)
        
        self.mar_history.append(raw_mar)
        smoothed_mar = np.mean(self.mar_history)
        
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return False, smoothed_mar, self._make_info(smoothed_mar, False, in_cooldown=True)
        
        mouth_open = smoothed_mar > self.mar_threshold
        self.yawn_window.append(1 if mouth_open else 0)
        open_count = sum(self.yawn_window)
        persistent_yawn = (open_count >= self.yawn_threshold)
        
        yawn_detected = False
        
        if persistent_yawn:
            self.current_yawn_frames += 1
            self.in_yawn = True
            self.peak_mar = max(self.peak_mar, smoothed_mar)
        else:
            if self.in_yawn and self.current_yawn_frames >= self.min_yawn_duration:
                self.yawn_count += 1
                self.yawn_timestamps.append(timestamp)
                self.cooldown_counter = self.cooldown_frames
                yawn_detected = True
                print(f"🥱 Yawn #{self.yawn_count} detected!")
            
            self.in_yawn = False
            self.current_yawn_frames = 0
            self.peak_mar = 0.0
        
        progress = min(self.current_yawn_frames / self.min_yawn_duration, 1.0) if self.in_yawn else 0.0
        
        return self.in_yawn, smoothed_mar, self._make_info(smoothed_mar, yawn_detected, 
                                                           progress=progress, open_count=open_count)
    
    def _make_info(self, mar, yawn_detected, in_cooldown=False, progress=0.0, open_count=0):
        return {
            'mar': mar,
            'yawn_count': self.yawn_count,
            'yawn_progress': progress,
            'in_cooldown': in_cooldown,
            'yawn_detected': yawn_detected,
            'yawns_per_minute': self._calc_frequency(),
            'in_yawn': self.in_yawn,
            'open_count': open_count
        }
    
    def _calc_frequency(self):
        if len(self.yawn_timestamps) < 2:
            duration = time.time() - self.session_start
            if duration < 60:
                return 0.0
            return self.yawn_count / (duration / 60.0)
        time_span = self.yawn_timestamps[-1] - self.yawn_timestamps[0]
        if time_span < 1:
            return 0.0
        return (len(self.yawn_timestamps) - 1) / (time_span / 60.0)
    
    def reset(self):
        self.yawn_window.clear()
        self.mar_history.clear()
        self.current_yawn_frames = 0
        self.yawn_count = 0
        self.cooldown_counter = 0
        self.in_yawn = False
        self.peak_mar = 0.0
        self.yawn_timestamps.clear()
        self.session_start = time.time()


# ============================================
# AUDIO ALARM SYSTEM
# ============================================

class AlarmController:
    """Manages audio alarm with hysteresis."""
    
    def __init__(self, sound_path, volume=0.7, enabled=True,
                trigger_threshold=5, min_duration_frames=60, stop_threshold=10,
                use_window=True, window_size=10):
        self.enabled = enabled
        self.sound = None
        self.is_playing = False
        self.trigger_threshold = trigger_threshold
        self.min_duration_frames = min_duration_frames
        self.stop_threshold = stop_threshold
        self.use_window = use_window
        self.window_size = window_size
        
        self.alarm_active = False
        self.drowsy_counter = 0
        self.awake_counter = 0
        self.frames_since_triggered = 0
        self.alarm_reason = ""  # Track why alarm triggered
        
        if use_window:
            self.drowsy_window = deque(maxlen=window_size)
        
        if not self.enabled:
            print("🔇 Audio alarm disabled")
            return
        
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        except Exception as e:
            print(f"❌ Could not initialize audio: {e}")
            self.enabled = False
            return
        
        try:
            self.sound = pygame.mixer.Sound(sound_path)
            self.sound.set_volume(volume)
            print(f"✅ Alarm loaded: {os.path.basename(sound_path)}")
        except FileNotFoundError:
            print(f"❌ Alarm not found: {sound_path}")
            self.enabled = False
        except Exception as e:
            print(f"❌ Error loading alarm: {e}")
            self.enabled = False
    
    def update_state(self, drowsy_alert, reason=""):
        if not self.enabled:
            return False
        
        if self.use_window:
            self.drowsy_window.append(1 if drowsy_alert else 0)
            drowsy_count = sum(self.drowsy_window)
            
            if drowsy_count >= self.trigger_threshold:
                if not self.alarm_active:
                    self.alarm_active = True
                    self.frames_since_triggered = 0
                    self.alarm_reason = reason
                    print(f"🚨 ALARM TRIGGERED: {reason}")
                self.awake_counter = 0
                self.frames_since_triggered += 1
            else:
                if self.alarm_active:
                    self.awake_counter += 1
                    self.frames_since_triggered += 1
                    if (self.frames_since_triggered >= self.min_duration_frames and
                        self.awake_counter >= self.stop_threshold):
                        self.alarm_active = False
                        self.alarm_reason = ""
                        print(f"✓ Alarm stopped")
        
        if self.alarm_active:
            self._play()
        else:
            self._stop()
        
        return self.alarm_active
    
    def trigger_immediate(self, reason=""):
        """Trigger alarm immediately without window check."""
        if not self.enabled:
            return False
        
        if not self.alarm_active:
            self.alarm_active = True
            self.frames_since_triggered = 0
            self.alarm_reason = reason
            self.drowsy_window.clear()
            for _ in range(self.window_size):
                self.drowsy_window.append(1)
            print(f"🚨 IMMEDIATE ALARM: {reason}")
        
        self._play()
        return True
    
    def _play(self):
        if self.sound and not self.is_playing:
            self.sound.play(loops=-1)
            self.is_playing = True
    
    def _stop(self):
        if self.sound and self.is_playing:
            self.sound.stop()
            self.is_playing = False
    
    def cleanup(self):
        if self.enabled:
            if self.sound:
                self.sound.stop()
            pygame.mixer.stop()
            pygame.mixer.quit()
            print("🔊 Audio cleaned up")


# ============================================
# OPTIMIZED MODEL LOADER
# ============================================

class OptimizedEyeModel:
    """Wrapper for eye state model with optimized inference."""
    
    def __init__(self, model_path, tflite_path=None, use_tflite=False):
        self.use_tflite = use_tflite and tflite_path and os.path.exists(tflite_path)
        
        if self.use_tflite:
            print("[INFO] Loading TFLite model...")
            self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print(f"[OK] TFLite loaded")
        else:
            print("[INFO] Loading Keras model...")
            self.model = self._load_keras_model(model_path)
            self._warmup()
            print(f"[OK] Keras loaded")
    
    def _load_keras_model(self, model_path):
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import (
            Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout,
            GlobalAveragePooling2D, Dense, Activation
        )
        from keras.regularizers import l2
        
        inputs = Input(shape=(64, 64, 1))
        
        x = Conv2D(32, (3, 3), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
        
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
        
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
        
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
        
        model = Model(inputs=inputs, outputs=outputs)
        model.load_weights(model_path)
        return model
    
    def _warmup(self):
        dummy = np.zeros((2, 64, 64, 1), dtype=np.float32)
        _ = self.model(dummy, training=False)
    
    def predict_batch(self, eye_images):
        if self.use_tflite:
            predictions = []
            for img in eye_images:
                self.interpreter.set_tensor(
                    self.input_details[0]['index'],
                    np.expand_dims(img, axis=0).astype(np.float32)
                )
                self.interpreter.invoke()
                pred = self.interpreter.get_tensor(self.output_details[0]['index'])
                predictions.append(pred[0][0])
            return np.array(predictions)
        else:
            predictions = self.model(eye_images, training=False)
            return predictions.numpy().flatten()


# ============================================
# PREPROCESSING
# ============================================

def extract_both_eyes_batch(frame, left_bbox, right_bbox):
    batch = np.zeros((2, 64, 64, 1), dtype=np.float32)
    valid = [False, False]
    
    for i, bbox in enumerate([left_bbox, right_bbox]):
        if bbox is None:
            continue
        x_min, y_min, x_max, y_max = bbox
        eye_crop = frame[y_min:y_max, x_min:x_max]
        if eye_crop.size == 0:
            continue
        eye_gray = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
        eye_clahe = clahe.apply(eye_gray)
        eye_resized = cv2.resize(eye_clahe, EYE_IMG_SIZE)
        batch[i, :, :, 0] = eye_resized.astype(np.float32) / 255.0
        valid[i] = True
    
    return batch, tuple(valid)


def get_eye_region(frame, landmarks, eye_indices, padding=EYE_BBOX_PADDING):
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
        
        eye_width = x_max - x_min
        eye_height = y_max - y_min
        
        pad_x = max(padding, int(eye_width * 0.3))
        pad_y = max(padding, int(eye_height * 0.5))
        
        x_min = max(0, x_min - pad_x)
        y_min = max(0, y_min - pad_y)
        x_max = min(w, x_max + pad_x)
        y_max = min(h, y_max + pad_y)
        
        return (x_min, y_min, x_max, y_max)
    return None


# ============================================
# PERFORMANCE MONITORING
# ============================================

class PerformanceMonitor:
    def __init__(self, window_size=30):
        self.timings = {
            'mediapipe': deque(maxlen=window_size),
            'preprocess': deque(maxlen=window_size),
            'inference': deque(maxlen=window_size),
            'fusion': deque(maxlen=window_size),
            'head_pose': deque(maxlen=window_size),
            'yawn': deque(maxlen=window_size),
            'display': deque(maxlen=window_size),
            'total': deque(maxlen=window_size)
        }
    
    def log(self, component, time_ms):
        self.timings[component].append(time_ms)
    
    def get_avg(self, component):
        if len(self.timings[component]) == 0:
            return 0
        return np.mean(self.timings[component])
    
    def get_summary(self):
        return {k: self.get_avg(k) for k in self.timings}


# ============================================
# DISPLAY FUNCTIONS
# ============================================

def draw_landmarks_custom(frame, landmarks, indices, color=(0, 255, 0), thickness=1):
    h, w, _ = frame.shape
    points = []
    
    for idx in indices:
        if idx < len(landmarks):
            landmark = landmarks[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            points.append((x, y))
            cv2.circle(frame, (x, y), 2, color, -1)
    
    if len(points) > 1:
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], color, thickness)
        if len(points) > 2:
            cv2.line(frame, points[-1], points[0], color, thickness)
    
    return points


def draw_mouth_landmarks(frame, landmarks, color=(0, 255, 255)):
    h, w = frame.shape[:2]
    points = []
    
    for idx in MOUTH_OUTER:
        try:
            lm = landmarks[idx]
            x = int(lm.x * w)
            y = int(lm.y * h)
            points.append((x, y))
            cv2.circle(frame, (x, y), 1, color, -1)
        except:
            continue
    
    if len(points) > 1:
        for i in range(len(points)):
            cv2.line(frame, points[i], points[(i+1) % len(points)], color, 1)


def display_info_v52(frame, fps, detection_status,
                     left_result=None, right_result=None,
                     drowsy_frames=0, drowsy_alert=False, alarm_active=False,
                     yawn_state=None, head_pose=None, head_droop_info=None,
                     perf_monitor=None, fatigue_level="NORMAL", alarm_reason=""):
    """Display status with metrics in corners - no blocking panel."""
    h, w = frame.shape[:2]

    # Add semi-transparent background for text readability
    def add_text_bg(frame, text, pos, font, scale, color, thickness, bg_color=(0, 0, 0), alpha=0.6):
        (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
        x, y = pos
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 5, y - text_h - 5), (x + text_w + 5, y + 5), bg_color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.putText(frame, text, pos, font, scale, color, thickness)

    # TOP LEFT CORNER - FPS and Status
    y_top = 25
    fps_color = (0, 255, 0) if fps >= 25 else (0, 165, 255) if fps >= 20 else (0, 0, 255)
    add_text_bg(frame, f"FPS: {fps:.1f}", (10, y_top), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)

    status_color = (0, 255, 0) if detection_status else (0, 0, 255)
    status_text = "Face OK" if detection_status else "No Face"
    add_text_bg(frame, status_text, (10, y_top + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

    # TOP RIGHT CORNER - Eye states
    if left_result and right_result:
        left_state = left_result.state.value
        left_color = (0, 0, 255) if left_state == "CLOSED" else (0, 165, 255) if left_state == "UNCERTAIN" else (0, 255, 0)
        add_text_bg(frame, f"L: {left_state} ({left_result.confidence:.0f}%)",
                   (w - 200, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 2)

        right_state = right_result.state.value
        right_color = (0, 0, 255) if right_state == "CLOSED" else (0, 165, 255) if right_state == "UNCERTAIN" else (0, 255, 0)
        add_text_bg(frame, f"R: {right_state} ({right_result.confidence:.0f}%)",
                   (w - 200, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 2)

        # EAR values
        add_text_bg(frame, f"EAR: {left_result.ear_value:.2f} / {right_result.ear_value:.2f}",
                   (w - 200, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # BOTTOM LEFT CORNER - Head Pose
    y_bottom = h - 80
    if head_pose and head_pose.confidence > 0.3:
        if head_pose.is_severely_down:
            hp_color = (0, 0, 255)
            hp_state = "DROOP!"
        elif head_pose.is_looking_down:
            hp_color = (0, 165, 255)
            hp_state = "Down"
        elif head_pose.is_looking_away:
            hp_color = (0, 255, 255)
            hp_state = "Away"
        else:
            hp_color = (0, 255, 0)
            hp_state = "Normal"

        add_text_bg(frame, f"Head: {hp_state}", (10, y_bottom),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, hp_color, 2)
        add_text_bg(frame, f"P:{head_pose.pitch:+.0f} Y:{head_pose.yaw:+.0f} R:{head_pose.roll:+.0f}",
                   (10, y_bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Head droop progress
        if head_droop_info:
            is_drooping, is_severe, droop_frames = head_droop_info
            if droop_frames > 0:
                droop_progress = min(droop_frames / HEAD_DROOP_ALARM_FRAMES, 1.0)
                bar_color = (0, 0, 255) if droop_progress > 0.8 else (0, 165, 255) if droop_progress > 0.5 else (0, 255, 255)
                bar_width = int(150 * droop_progress)
                cv2.rectangle(frame, (10, y_bottom + 35), (160, y_bottom + 48), (50, 50, 50), -1)
                if bar_width > 0:
                    cv2.rectangle(frame, (10, y_bottom + 35), (10 + bar_width, y_bottom + 48), bar_color, -1)
                add_text_bg(frame, f"{droop_frames}/{HEAD_DROOP_ALARM_FRAMES}",
                           (165, y_bottom + 47), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    # BOTTOM RIGHT CORNER - Yawn and Drowsiness
    y_bottom_right = h - 80
    if yawn_state:
        mar = yawn_state.get('mar', 0)
        in_yawn = yawn_state.get('in_yawn', False)
        yawn_count = yawn_state.get('yawn_count', 0)

        yawn_color = (0, 0, 255) if in_yawn else (0, 255, 0)
        yawn_text = "YAWNING" if in_yawn else "Mouth OK"
        add_text_bg(frame, f"{yawn_text}", (w - 180, y_bottom_right),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, yawn_color, 2)
        add_text_bg(frame, f"MAR: {mar:.2f} | Yawns: {yawn_count}",
                   (w - 220, y_bottom_right + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # Drowsiness progress bar at bottom center
    if left_result and right_result:
        threshold = DROWSINESS_FRAME_THRESHOLD
        progress = min(drowsy_frames / threshold, 1.0)
        bar_width = int(200 * progress)
        bar_color = (0, 255, 0) if progress < 0.5 else (0, 165, 255) if progress < 0.8 else (0, 0, 255)
        bar_x = w // 2 - 100
        bar_y = h - 30
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 200, bar_y + 15), (50, 50, 50), -1)
        if bar_width > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 15), bar_color, -1)
        add_text_bg(frame, f"Drowsy: {drowsy_frames}/{threshold}",
                   (bar_x + 210, bar_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # Performance info (if enabled)
    if perf_monitor:
        summary = perf_monitor.get_summary()
        perf_text = f"MP:{summary['mediapipe']:.0f} CNN:{summary['inference']:.0f} HP:{summary['head_pose']:.0f}ms"
        add_text_bg(frame, perf_text, (10, y_top + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # Instructions at very bottom
    add_text_bg(frame, "q:quit | p:perf | h:head_viz", (10, h - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    # Alarm overlay
    if alarm_active:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # Show alarm reason
        alarm_text = alarm_reason if alarm_reason else "DROWSINESS DETECTED"
        cv2.putText(frame, f"!!! {alarm_text} !!!", (w//2 - 220, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    elif yawn_state and yawn_state.get('in_yawn', False):
        cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 200, 255), 4)
    elif fatigue_level == "MODERATE":
        cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 165, 255), 3)


def load_eye_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, MODEL_PATH)
    tflite_path = os.path.join(script_dir, TFLITE_PATH)
    
    if USE_TFLITE and os.path.exists(tflite_path):
        return OptimizedEyeModel(model_path, tflite_path, use_tflite=True)
    
    if os.path.exists(model_path):
        return OptimizedEyeModel(model_path, tflite_path, use_tflite=False)
    
    print(f"[WARNING] Model not found at {model_path}")
    return None


# ============================================
# CONSOLE LOGGING UTILITIES (NEW FOR DEMO)
# ============================================

class ConsoleLogger:
    """
    Enhanced console logging for real-time monitoring during demo.
    
    Outputs detailed per-frame diagnostics including:
    - Head pose angles and state
    - Eye detection states and confidence
    - Yawn detection status
    - Alarm triggers with reasons
    - Performance metrics
    """
    
    def __init__(self):
        self.frame_count = 0
        self.last_head_state = None
        self.last_alarm_reason = None
        
    def log_frame(self, fps, head_pose, head_droop_info, left_result, right_result,
                  yawn_state, drowsy_frames, alarm_active, alarm_reason, perf_monitor=None):
        """Log comprehensive frame information to console."""
        self.frame_count += 1
        
        # Check if we should log this frame
        log_periodic = (self.frame_count % CONSOLE_LOG_EVERY_N_FRAMES == 0) if ENABLE_CONSOLE_LOGGING else False
        
        # Determine head state for change detection
        head_state = "Unknown"
        if head_pose and head_pose.confidence > 0.3:
            if head_pose.is_severely_down:
                head_state = "DROOP!"
            elif head_pose.is_looking_down:
                head_state = "Down"
            elif head_pose.is_looking_away:
                head_state = "Away"
            else:
                head_state = "Normal"
        
        # Check for state change
        state_changed = (head_state != self.last_head_state) and CONSOLE_LOG_STATE_CHANGES
        
        # Check for alarm trigger
        alarm_triggered = alarm_active and (alarm_reason != self.last_alarm_reason) and CONSOLE_LOG_ALARMS
        
        # Update state tracking
        self.last_head_state = head_state
        self.last_alarm_reason = alarm_reason if alarm_active else None
        
        # Log if any condition is met
        if log_periodic or state_changed or alarm_triggered:
            self._print_frame_info(fps, head_pose, head_state, head_droop_info,
                                   left_result, right_result, yawn_state,
                                   drowsy_frames, alarm_active, alarm_reason,
                                   state_changed, alarm_triggered, perf_monitor)
    
    def _print_frame_info(self, fps, head_pose, head_state, head_droop_info,
                          left_result, right_result, yawn_state, drowsy_frames,
                          alarm_active, alarm_reason, state_changed, alarm_triggered,
                          perf_monitor):
        """Print formatted frame information."""
        # Frame header
        print(f"\n=== FRAME {self.frame_count} @ {fps:.1f} FPS ===")
        
        # Head pose line
        if head_pose and head_pose.confidence > 0.1:
            p, y, r = head_pose.pitch, head_pose.yaw, head_pose.roll
            conf = head_pose.confidence
            state_marker = "*" if state_changed else ""
            print(f"Head Pose: P:{p:+.0f}° Y:{y:+.0f}° R:{r:+.0f}° | Conf: {conf:.2f} | State: {head_state}{state_marker}")
            
            # Droop progress if applicable
            if head_droop_info:
                _, is_severe, droop_frames = head_droop_info
                if droop_frames > 0:
                    print(f"   Head Droop: {droop_frames}/{HEAD_DROOP_ALARM_FRAMES} frames {'(SEVERE!)' if is_severe else ''}")
        else:
            print("Head Pose: No face detected")
        
        # Eye state line
        if left_result and right_result:
            l_state = left_result.state.name if hasattr(left_result.state, 'name') else str(left_result.state)
            r_state = right_result.state.name if hasattr(right_result.state, 'name') else str(right_result.state)
            l_conf = left_result.confidence * 100
            r_conf = right_result.confidence * 100
            l_ear = getattr(left_result, 'ear', 0)
            r_ear = getattr(right_result, 'ear', 0)
            print(f"Eye State: L:{l_state}({l_conf:.0f}%) R:{r_state}({r_conf:.0f}%) | EAR: {l_ear:.2f}/{r_ear:.2f}")
        
        # Yawn line
        if yawn_state:
            mar = yawn_state.get('mar', 0)
            in_yawn = yawn_state.get('in_yawn', False)
            yawn_count = yawn_state.get('yawn_count', 0)
            yawn_text = "YAWNING" if in_yawn else "OK"
            print(f"Yawn: Mouth {yawn_text} | MAR: {mar:.2f} | Count: {yawn_count}")
        
        # Drowsy frames
        print(f"Drowsy Frames: {drowsy_frames}/{DROWSINESS_FRAME_THRESHOLD}")
        
        # Alarm trigger
        if alarm_triggered and alarm_active:
            print(f"⚠️  ALARM TRIGGERED: {alarm_reason}")
        elif alarm_active:
            print(f"🔔 ALARM ACTIVE: {alarm_reason}")
        
        # Performance if available
        if perf_monitor:
            summary = perf_monitor.get_summary()
            mp = summary.get('mediapipe', 0)
            cnn = summary.get('inference', 0)
            hp = summary.get('head_pose', 0)
            total = summary.get('total', 0)
            print(f"Performance: MP:{mp:.0f}ms CNN:{cnn:.0f}ms HP:{hp:.0f}ms Total:{total:.0f}ms")


# Global console logger instance
console_logger = ConsoleLogger()


# ============================================
# MAIN FUNCTION
# ============================================

def main():
    global SKIP_FRAME_MODE, YAWN_MAR_THRESHOLD, SHOW_HEAD_POSE_VISUALIZATION, ENABLE_CONSOLE_LOGGING
    
    print("=" * 70)
    print("DRIVER DROWSINESS DETECTION SYSTEM v5.2")
    print("With Head Pose Estimation")
    print("=" * 70)
    
    # Load model
    eye_model = load_eye_model()
    
    # Initialize camera
    print("\n[INIT] Starting camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return
    
    print("[OK] Camera initialized")
    
    # Initialize alarm
    alarm = AlarmController(
        sound_path=ALARM_SOUND_PATH,
        volume=ALARM_VOLUME,
        enabled=ENABLE_AUDIO,
        use_window=True,
        window_size=10,
        trigger_threshold=5,
        min_duration_frames=30,
        stop_threshold=10
    )
    
    # Initialize ENHANCED eye detector (replaces TemporalSmoother)
    left_eye_detector = EnhancedEyeStateDetector()
    right_eye_detector = EnhancedEyeStateDetector()
    print("[OK] Enhanced eye detectors initialized (CNN+EAR fusion)")
    
    # Initialize HEAD POSE ESTIMATOR (NEW)
    head_pose_estimator = HeadPoseEstimator(
        smoothing_alpha=HEAD_POSE_SMOOTHING,
        looking_down_threshold=HEAD_LOOKING_DOWN_THRESHOLD,
        severe_droop_threshold=HEAD_SEVERE_DROOP_THRESHOLD,
        looking_away_threshold=HEAD_LOOKING_AWAY_THRESHOLD,
    )
    print(f"[OK] Head pose estimator initialized (thresholds: down={HEAD_LOOKING_DOWN_THRESHOLD}°, droop={HEAD_SEVERE_DROOP_THRESHOLD}°)")
    
    # Track closed frames for drowsiness
    both_closed_frames = 0
    recent_yawn_frames = deque(maxlen=YAWN_RECENT_WINDOW)
    
    # Initialize yawn detector
    yawn_detector = YawnDetectorMAR(
        mar_threshold=YAWN_MAR_THRESHOLD,
        temporal_window=YAWN_TEMPORAL_WINDOW,
        yawn_threshold=YAWN_THRESHOLD_FRAMES,
        min_yawn_duration=YAWN_MIN_DURATION,
        cooldown_frames=YAWN_COOLDOWN_FRAMES
    )
    print(f"[OK] Yawn detector initialized")
    
    # Performance monitor
    perf_monitor = PerformanceMonitor()
    
    frame_count = 0
    last_cnn_predictions = (0.5, 0.5)
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        print("[OK] MediaPipe FaceMesh initialized")
        print(f"\nStarting detection...")
        print("Press 'q' to quit, 'p' for perf stats, 'h' to toggle head pose visualization\n")
        
        prev_time = time.time()
        show_perf = False
        
        while True:
            frame_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # MediaPipe processing
            mp_start = time.time()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            perf_monitor.log('mediapipe', (time.time() - mp_start) * 1000)
            
            face_detected = False
            left_result = None
            right_result = None
            drowsy_alert = False
            alarm_active = False
            yawn_state = None
            is_yawning = False
            head_pose = None
            head_droop_info = None
            alarm_reason = ""
            
            if results.multi_face_landmarks:
                face_detected = True
                
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = face_landmarks.landmark
                    
                    # Draw eye landmarks
                    draw_landmarks_custom(frame, landmarks, LEFT_EYE_INDICES, (0, 255, 0), 1)
                    draw_landmarks_custom(frame, landmarks, RIGHT_EYE_INDICES, (0, 255, 0), 1)
                    
                    # Draw mouth landmarks
                    yawn_color = (0, 0, 255) if is_yawning else (0, 255, 255)
                    draw_mouth_landmarks(frame, landmarks, yawn_color)
                    
                    # ============================================
                    # HEAD POSE ESTIMATION (NEW)
                    # ============================================
                    hp_start = time.time()
                    head_pose = head_pose_estimator.update(landmarks, w, h)
                    perf_monitor.log('head_pose', (time.time() - hp_start) * 1000)
                    
                    # Check for head drooping (microsleep indicator)
                    head_droop_info = head_pose_estimator.detect_head_drooping(
                        threshold_frames=HEAD_DROOP_ALARM_FRAMES,
                        severe_frames=HEAD_SEVERE_DROOP_FRAMES,
                    )
                    is_head_drooping, is_severe_droop, droop_frames = head_droop_info
                    
                    # Draw head pose visualization if enabled
                    if SHOW_HEAD_POSE_VISUALIZATION and head_pose.confidence > 0.5:
                        nose_lm = landmarks[1]  # Nose tip
                        nose_point = (int(nose_lm.x * w), int(nose_lm.y * h))
                        draw_head_pose_axis(frame, head_pose, nose_point, axis_length=50)
                    
                    # Get eye bounding boxes
                    left_bbox = get_eye_region(frame, landmarks, LEFT_EYE_INDICES)
                    right_bbox = get_eye_region(frame, landmarks, RIGHT_EYE_INDICES)
                    
                    # Calculate EAR for both eyes
                    left_ear = calculate_ear(landmarks, LEFT_EYE_INDICES)
                    right_ear = calculate_ear(landmarks, RIGHT_EYE_INDICES)
                    
                    # Eye detection with CNN
                    if eye_model is not None:
                        run_cnn = True
                        if SKIP_FRAME_MODE and frame_count % CNN_EVERY_N_FRAMES != 0:
                            run_cnn = False
                        
                        if run_cnn:
                            preprocess_start = time.time()
                            batch, valid = extract_both_eyes_batch(frame, left_bbox, right_bbox)
                            perf_monitor.log('preprocess', (time.time() - preprocess_start) * 1000)
                            
                            inference_start = time.time()
                            if valid[0] or valid[1]:
                                predictions = eye_model.predict_batch(batch)
                                left_cnn = predictions[0] if valid[0] else 0.5
                                right_cnn = predictions[1] if valid[1] else 0.5
                                last_cnn_predictions = (left_cnn, right_cnn)
                            perf_monitor.log('inference', (time.time() - inference_start) * 1000)
                        else:
                            left_cnn, right_cnn = last_cnn_predictions
                        
                        # ENHANCED DETECTION with fusion + HEAD POSE
                        fusion_start = time.time()
                        
                        # Pass head pitch to eye detectors for fusion adjustment
                        head_pitch = head_pose.pitch if head_pose and head_pose.confidence > 0.3 else None
                        
                        left_result = left_eye_detector.update(left_cnn, left_ear, head_pitch=head_pitch)
                        right_result = right_eye_detector.update(right_cnn, right_ear, head_pitch=head_pitch)
                        perf_monitor.log('fusion', (time.time() - fusion_start) * 1000)
                        
                        # Draw eye bounding boxes based on state
                        if left_bbox:
                            color = (0, 0, 255) if left_result.state == EyeState.CLOSED else \
                                    (0, 165, 255) if left_result.state == EyeState.UNCERTAIN else (0, 255, 0)
                            cv2.rectangle(frame, (left_bbox[0], left_bbox[1]),
                                        (left_bbox[2], left_bbox[3]), color, 2)
                        
                        if right_bbox:
                            color = (0, 0, 255) if right_result.state == EyeState.CLOSED else \
                                    (0, 165, 255) if right_result.state == EyeState.UNCERTAIN else (0, 255, 0)
                            cv2.rectangle(frame, (right_bbox[0], right_bbox[1]),
                                        (right_bbox[2], right_bbox[3]), color, 2)
                        
                        # Update drowsy frame counter
                        left_closed = left_result.state == EyeState.CLOSED
                        right_closed = right_result.state == EyeState.CLOSED
                        left_uncertain = left_result.state == EyeState.UNCERTAIN
                        right_uncertain = right_result.state == EyeState.UNCERTAIN
                        
                        if left_closed and right_closed:
                            both_closed_frames += 1
                        elif (left_closed and right_uncertain) or (right_closed and left_uncertain):
                            both_closed_frames += 0.75
                        elif left_uncertain and right_uncertain:
                            both_closed_frames += 0.5
                        else:
                            both_closed_frames = max(0, both_closed_frames - 2)
                    
                    # YAWN DETECTION
                    yawn_start = time.time()
                    is_yawning, mar, yawn_state = yawn_detector.update(landmarks, h, w)
                    perf_monitor.log('yawn', (time.time() - yawn_start) * 1000)
                    
                    # Track recent yawns for faster eye closure trigger
                    recent_yawn_frames.append(1 if is_yawning else 0)
            
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time
            
            # ============================================
            # DROWSINESS DECISION LOGIC (UPDATED)
            # ============================================
            
            # Check for SEVERE HEAD DROOP - immediate alarm
            if head_droop_info and head_droop_info[1]:  # is_severe_droop
                alarm_reason = "MICROSLEEP - HEAD DROOP"
                alarm.trigger_immediate(alarm_reason)
                drowsy_alert = True
                alarm_active = alarm.alarm_active
            else:
                # Determine drowsiness threshold (lower if recent yawn)
                recent_yawn_activity = sum(recent_yawn_frames) > 0 if recent_yawn_frames else False
                threshold = max(10, DROWSINESS_FRAME_THRESHOLD // 2) if recent_yawn_activity else DROWSINESS_FRAME_THRESHOLD
                
                # Check for yawn-based fatigue
                yawn_count = yawn_state.get('yawn_count', 0) if yawn_state else 0
                yawn_freq_windowed = 0.0
                if yawn_state and yawn_detector.yawn_timestamps:
                    recent_yawns = [t for t in yawn_detector.yawn_timestamps
                                   if current_time - t <= YAWN_OBSERVATION_WINDOW]
                    if len(recent_yawns) >= 2:
                        time_span = recent_yawns[-1] - recent_yawns[0]
                        if time_span >= 30:
                            yawn_freq_windowed = len(recent_yawns) / (time_span / 60.0)
                
                extreme_yawn_fatigue = (yawn_count >= YAWN_MIN_COUNT_FOR_ALARM and 
                                        yawn_freq_windowed >= YAWN_DANGER_THRESHOLD)
                
                # Check for head drooping + eyes closed combination
                head_droop_with_eyes = False
                if head_droop_info and head_droop_info[0]:  # is_head_drooping
                    if left_result and right_result:
                        if left_result.state == EyeState.CLOSED or right_result.state == EyeState.CLOSED:
                            head_droop_with_eyes = True
                            alarm_reason = "HEAD DROOP + EYES CLOSED"
                
                # Combine signals
                eye_drowsy = both_closed_frames >= threshold
                
                if eye_drowsy:
                    alarm_reason = "EYES CLOSED"
                elif extreme_yawn_fatigue:
                    alarm_reason = "EXCESSIVE YAWNING"
                elif head_droop_with_eyes:
                    pass  # already set above
                
                drowsy_alert = eye_drowsy or extreme_yawn_fatigue or head_droop_with_eyes
                alarm_active = alarm.update_state(drowsy_alert, alarm_reason)
            
            # Fatigue level
            if alarm_active:
                fatigue_level = "CRITICAL"
            elif head_droop_info and head_droop_info[2] > 30:  # droop_frames > 30
                fatigue_level = "MODERATE"
            elif yawn_state:
                yawn_count = yawn_state.get('yawn_count', 0)
                yawn_freq = yawn_state.get('yawns_per_minute', 0)
                if yawn_count >= 3 and yawn_freq >= YAWN_FATIGUE_THRESHOLD:
                    fatigue_level = "MODERATE"
                elif is_yawning:
                    fatigue_level = "MILD"
                else:
                    fatigue_level = "NORMAL"
            else:
                fatigue_level = "NORMAL"
            
            # Display
            display_start = time.time()
            display_info_v52(frame, fps, face_detected, left_result, right_result,
                            int(both_closed_frames), drowsy_alert, alarm_active,
                            yawn_state, head_pose, head_droop_info,
                            perf_monitor if show_perf else None, fatigue_level, 
                            alarm.alarm_reason if alarm_active else "")
            perf_monitor.log('display', (time.time() - display_start) * 1000)
            perf_monitor.log('total', (time.time() - frame_start) * 1000)
            
            # CONSOLE LOGGING (NEW FOR DEMO)
            console_logger.log_frame(
                fps=fps,
                head_pose=head_pose,
                head_droop_info=head_droop_info,
                left_result=left_result,
                right_result=right_result,
                yawn_state=yawn_state,
                drowsy_frames=int(both_closed_frames),
                alarm_active=alarm_active,
                alarm_reason=alarm.alarm_reason if alarm_active else "",
                perf_monitor=perf_monitor if show_perf else None
            )
            
            cv2.imshow('Drowsiness Detection v5.2 (Head Pose)', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[STOPPING] Terminated by user")
                break
            elif key == ord('p'):
                show_perf = not show_perf
                print(f"[INFO] Performance display: {'ON' if show_perf else 'OFF'}")
            elif key == ord('s'):
                SKIP_FRAME_MODE = not SKIP_FRAME_MODE
                print(f"[INFO] Skip-frame mode: {'ON' if SKIP_FRAME_MODE else 'OFF'}")
            elif key == ord('h'):
                SHOW_HEAD_POSE_VISUALIZATION = not SHOW_HEAD_POSE_VISUALIZATION
                print(f"[INFO] Head pose visualization: {'ON' if SHOW_HEAD_POSE_VISUALIZATION else 'OFF'}")
            elif key == ord('l'):
                global ENABLE_CONSOLE_LOGGING
                ENABLE_CONSOLE_LOGGING = not ENABLE_CONSOLE_LOGGING
                print(f"[INFO] Console logging: {'ON' if ENABLE_CONSOLE_LOGGING else 'OFF'}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    alarm.cleanup()
    
    # Final report
    print("\n" + "=" * 70)
    print("SESSION SUMMARY")
    print("=" * 70)
    summary = perf_monitor.get_summary()
    print(f"Average timings:")
    print(f"  MediaPipe:  {summary['mediapipe']:.1f}ms")
    print(f"  Preprocess: {summary['preprocess']:.1f}ms")
    print(f"  Eye CNN:    {summary['inference']:.1f}ms")
    print(f"  Fusion:     {summary['fusion']:.1f}ms")
    print(f"  Head Pose:  {summary['head_pose']:.1f}ms")
    print(f"  Yawn MAR:   {summary['yawn']:.1f}ms")
    print(f"  Display:    {summary['display']:.1f}ms")
    print(f"  Total:      {summary['total']:.1f}ms ({1000/max(0.001, summary['total']):.1f} FPS)")
    
    # Eye detector stats
    print(f"\nEye Detection Stats:")
    left_stats = left_eye_detector.get_stats()
    print(f"  Left eye overrides: {left_stats['override_count']} ({left_stats['override_rate']*100:.1f}%)")
    right_stats = right_eye_detector.get_stats()
    print(f"  Right eye overrides: {right_stats['override_count']} ({right_stats['override_rate']*100:.1f}%)")
    
    # Head pose stats
    print(f"\nHead Pose Stats:")
    hp_stats = head_pose_estimator.get_stats()
    print(f"  Average time: {hp_stats['avg_time_ms']:.2f}ms")
    print(f"  Pitch range: {hp_stats['pitch_range']}")
    print(f"  Looking down events: {hp_stats['looking_down']}")
    
    print(f"\nYawn Statistics:")
    print(f"  Total yawns: {yawn_detector.yawn_count}")
    print("=" * 70)


if __name__ == "__main__":
    main()
