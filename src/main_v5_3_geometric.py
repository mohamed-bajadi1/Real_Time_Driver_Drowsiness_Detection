"""
Real-Time Driver Drowsiness Detection System
Main Application v5.3 - GEOMETRIC HEAD DROOP DETECTION

=====================================================================
MAJOR CHANGES FROM v5.2:
=====================================================================
1. REPLACED PnP-based head pose with GEOMETRIC head droop detection
2. Detects sideways head drooping (left/right) which PnP CANNOT
3. Uses eye line tilt + face vertical drop for robust detection
4. Much faster (<1ms vs ~3ms for PnP)

WHY THIS CHANGE:
- PnP Euler angle extraction fails for sideways head droop
- When head falls left/right, most rotation goes to ROLL, not PITCH
- Our threshold was on PITCH, so sideways droop was never detected
- The new geometric approach directly measures tilt and drop

DETECTION SIGNALS:
- Eye line tilt angle: When head tilts sideways, eyes tilt
- Face vertical drop: When head droops, face moves down in frame
- Both signals combined catch ALL droop directions

=====================================================================
Author: Binomial Team (Sprint 5 - Bug Fix)
Date: January 2026
=====================================================================
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

# Import NEW geometric head droop detector
from geometric_head_droop_detector import (
    GeometricHeadDroopDetector,
    GeometricHeadResult,
    HeadState,
    draw_geometric_head_info,
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
ALARM_SOUND_PATH = os.path.join(PROJECT_ROOT, "data", "sounds", "alarm.wav")

# Camera settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Model settings
MODEL_PATH = "../models/eye_state_classifier_v4.h5"
TFLITE_PATH = "../models/eye_state_classifier_v4.tflite"
USE_TFLITE = True

EYE_IMG_SIZE = (64, 64)
DETECTION_THRESHOLD = 0.4

# Drowsiness settings
DROWSINESS_FRAME_THRESHOLD = 20
ALARM_COOLDOWN_FRAMES = 60

# Yawn settings
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
# GEOMETRIC HEAD DROOP CONFIGURATION (NEW)
# ============================================
# Eye tilt thresholds (degrees) - Increased to reduce false positives
HEAD_TILT_WARNING = 16.0     # Start warning at 16° tilt (was 12°)
HEAD_TILT_DANGER = 24.0      # Danger at 24° tilt (was 20°)

# Vertical drop thresholds (ratio of face height) - Increased to reduce false positives
HEAD_DROP_WARNING = 0.08     # 8% face height drop (was 6%)
HEAD_DROP_DANGER = 0.14      # 14% face height drop (was 12%)

# Droop frame thresholds - Increased for more sustained detection before alerting
HEAD_DROOP_WARNING_FRAMES = 60   # ~2.0 seconds (was 1.5s)
HEAD_DROOP_DANGER_FRAMES = 35    # ~1.2 seconds (was 0.8s)

# Yaw threshold for "looking away"
HEAD_YAW_AWAY_THRESHOLD = 35.0

# Visualization
SHOW_HEAD_DEBUG = False      # Show detailed scores

# Console logging
ENABLE_CONSOLE_LOGGING = True
CONSOLE_LOG_EVERY_N_FRAMES = 30
CONSOLE_LOG_STATE_CHANGES = True

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
MOUTH_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
               409, 270, 269, 267, 0, 37, 39, 40, 185]

clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_SIZE)


# ============================================
# YAWN DETECTOR (unchanged)
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
# ALARM CONTROLLER (unchanged)
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
        self.alarm_reason = ""
        
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
        """Trigger alarm immediately."""
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
# MODEL LOADER (unchanged)
# ============================================

class OptimizedEyeModel:
    """Wrapper for eye state model."""
    
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
# PREPROCESSING (unchanged)
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
            'head_droop': deque(maxlen=window_size),
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
# DISPLAY
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


def display_info_v53(frame, fps, detection_status,
                     left_result=None, right_result=None,
                     drowsy_frames=0, drowsy_alert=False, alarm_active=False,
                     yawn_state=None, head_result=None,
                     perf_monitor=None, alarm_reason=""):
    """Display status with geometric head droop info."""
    h, w = frame.shape[:2]

    def add_text_bg(frame, text, pos, font, scale, color, thickness, bg_color=(0, 0, 0), alpha=0.6):
        (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
        x, y = pos
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 5, y - text_h - 5), (x + text_w + 5, y + 5), bg_color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.putText(frame, text, pos, font, scale, color, thickness)

    # TOP LEFT - FPS and Status
    y_top = 25
    fps_color = (0, 255, 0) if fps >= 25 else (0, 165, 255) if fps >= 20 else (0, 0, 255)
    add_text_bg(frame, f"FPS: {fps:.1f}", (10, y_top), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)

    status_color = (0, 255, 0) if detection_status else (0, 0, 255)
    status_text = "Face OK" if detection_status else "No Face"
    add_text_bg(frame, status_text, (10, y_top + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

    # TOP RIGHT - Eye states
    if left_result and right_result:
        left_state = left_result.state.value
        left_color = (0, 0, 255) if left_state == "CLOSED" else (0, 165, 255) if left_state == "UNCERTAIN" else (0, 255, 0)
        add_text_bg(frame, f"L: {left_state} ({left_result.confidence:.0f}%)",
                   (w - 200, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 2)

        right_state = right_result.state.value
        right_color = (0, 0, 255) if right_state == "CLOSED" else (0, 165, 255) if right_state == "UNCERTAIN" else (0, 255, 0)
        add_text_bg(frame, f"R: {right_state} ({right_result.confidence:.0f}%)",
                   (w - 200, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 2)

        add_text_bg(frame, f"EAR: {left_result.ear_value:.2f} / {right_result.ear_value:.2f}",
                   (w - 200, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # BOTTOM LEFT - Head Droop (NEW)
    y_bottom = h - 80
    if head_result and head_result.confidence > 0.3:
        # State color
        state_colors = {
            HeadState.NORMAL: (0, 255, 0),
            HeadState.LOOKING_DOWN: (0, 165, 255),
            HeadState.LOOKING_AWAY: (0, 255, 255),
            HeadState.DROWSY: (0, 100, 255),
            HeadState.DROOP: (0, 0, 255),
            HeadState.CALIBRATING: (255, 255, 0),
        }
        hp_color = state_colors.get(head_result.state, (255, 255, 255))
        
        add_text_bg(frame, f"Head: {head_result.state.value}", (10, y_bottom),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, hp_color, 2)
        
        # Show key metrics
        tilt_str = f"Tilt:{head_result.eye_tilt_deg:+.0f}°"
        drop_str = f"Drop:{head_result.vertical_drop_ratio:.2f}"
        add_text_bg(frame, f"{tilt_str} {drop_str}", (10, y_bottom + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Droop progress bar
        if head_result.droop_frames > 0:
            droop_progress = min(head_result.droop_frames / HEAD_DROOP_WARNING_FRAMES, 1.0)
            bar_color = (0, 0, 255) if droop_progress > 0.8 else (0, 165, 255) if droop_progress > 0.5 else (0, 255, 255)
            bar_width = int(150 * droop_progress)
            cv2.rectangle(frame, (10, y_bottom + 35), (160, y_bottom + 48), (50, 50, 50), -1)
            if bar_width > 0:
                cv2.rectangle(frame, (10, y_bottom + 35), (10 + bar_width, y_bottom + 48), bar_color, -1)
            add_text_bg(frame, f"{head_result.droop_frames}/{HEAD_DROOP_WARNING_FRAMES}",
                       (165, y_bottom + 47), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    # BOTTOM RIGHT - Yawn
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

    # Drowsiness progress bar
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

    # Performance
    if perf_monitor:
        summary = perf_monitor.get_summary()
        perf_text = f"MP:{summary['mediapipe']:.0f} CNN:{summary['inference']:.0f} HD:{summary['head_droop']:.0f}ms"
        add_text_bg(frame, perf_text, (10, y_top + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # Instructions
    add_text_bg(frame, "q:quit | p:perf | d:debug", (10, h - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    # Alarm overlay
    if alarm_active:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        alarm_text = alarm_reason if alarm_reason else "DROWSINESS DETECTED"
        cv2.putText(frame, f"!!! {alarm_text} !!!", (w//2 - 220, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)


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
# CONSOLE LOGGER
# ============================================

class ConsoleLogger:
    """Console logging for demo monitoring."""
    
    def __init__(self):
        self.frame_count = 0
        self.last_state = None
        self.last_alarm = None
    
    def log(self, fps, head_result, left_result, right_result, yawn_state,
            drowsy_frames, alarm_active, alarm_reason, perf=None):
        self.frame_count += 1
        
        # Get current state
        curr_state = head_result.state.value if head_result else "Unknown"
        
        # Check for logging conditions
        periodic = (self.frame_count % CONSOLE_LOG_EVERY_N_FRAMES == 0) if ENABLE_CONSOLE_LOGGING else False
        state_changed = (curr_state != self.last_state) and CONSOLE_LOG_STATE_CHANGES
        alarm_new = alarm_active and (alarm_reason != self.last_alarm)
        
        self.last_state = curr_state
        self.last_alarm = alarm_reason if alarm_active else None
        
        if periodic or state_changed or alarm_new:
            self._print(fps, head_result, left_result, right_result, yawn_state,
                       drowsy_frames, alarm_active, alarm_reason, state_changed, alarm_new, perf)
    
    def _print(self, fps, head, left, right, yawn, drowsy, alarm, reason, state_ch, alarm_new, perf):
        print(f"\n=== FRAME {self.frame_count} @ {fps:.1f} FPS ===")
        
        if head and head.confidence > 0.1:
            state_mark = "*" if state_ch else ""
            print(f"Head: {head.state.value}{state_mark} | Tilt:{head.eye_tilt_deg:+.0f}° Drop:{head.vertical_drop_ratio:.2f}")
            print(f"      Score:{head.droop_score:.2f} Frames:{head.droop_frames}")
        
        if left and right:
            print(f"Eyes: L:{left.state.value}({left.confidence:.0f}%) R:{right.state.value}({right.confidence:.0f}%)")
        
        if yawn:
            print(f"Yawn: MAR:{yawn['mar']:.2f} Count:{yawn['yawn_count']}")
        
        print(f"Drowsy: {drowsy}/{DROWSINESS_FRAME_THRESHOLD}")
        
        if alarm_new:
            print(f"⚠️  ALARM: {reason}")
        elif alarm:
            print(f"🔔 ALARM ACTIVE: {reason}")


console = ConsoleLogger()


# ============================================
# MAIN
# ============================================

def main():
    global SKIP_FRAME_MODE, SHOW_HEAD_DEBUG, ENABLE_CONSOLE_LOGGING
    
    print("=" * 70)
    print("DRIVER DROWSINESS DETECTION SYSTEM v5.3")
    print("With GEOMETRIC Head Droop Detection (Sideways Support)")
    print("=" * 70)
    
    eye_model = load_eye_model()
    
    print("\n[INIT] Starting camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return
    print("[OK] Camera initialized")
    
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
    
    left_eye_detector = EnhancedEyeStateDetector()
    right_eye_detector = EnhancedEyeStateDetector()
    print("[OK] Enhanced eye detectors initialized")
    
    # NEW: Geometric head droop detector
    head_detector = GeometricHeadDroopDetector(
        tilt_warning_deg=HEAD_TILT_WARNING,
        tilt_danger_deg=HEAD_TILT_DANGER,
        drop_warning_ratio=HEAD_DROP_WARNING,
        drop_danger_ratio=HEAD_DROP_DANGER,
        yaw_away_threshold=HEAD_YAW_AWAY_THRESHOLD,
        droop_warning_frames=HEAD_DROOP_WARNING_FRAMES,
        droop_danger_frames=HEAD_DROOP_DANGER_FRAMES,
        calibration_frames=30,
    )
    print(f"[OK] Geometric head droop detector initialized")
    print(f"     Tilt: warn={HEAD_TILT_WARNING}° danger={HEAD_TILT_DANGER}°")
    print(f"     Drop: warn={HEAD_DROP_WARNING:.0%} danger={HEAD_DROP_DANGER:.0%}")
    
    both_closed_frames = 0
    recent_yawn_frames = deque(maxlen=YAWN_RECENT_WINDOW)
    
    yawn_detector = YawnDetectorMAR(
        mar_threshold=YAWN_MAR_THRESHOLD,
        temporal_window=YAWN_TEMPORAL_WINDOW,
        yawn_threshold=YAWN_THRESHOLD_FRAMES,
        min_yawn_duration=YAWN_MIN_DURATION,
        cooldown_frames=YAWN_COOLDOWN_FRAMES
    )
    print("[OK] Yawn detector initialized")
    
    perf = PerformanceMonitor()
    frame_count = 0
    last_cnn = (0.5, 0.5)
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        print("[OK] MediaPipe FaceMesh initialized")
        print(f"\nStarting detection...")
        print("Press 'q' to quit, 'p' for perf, 'd' for debug\n")
        
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
            
            # MediaPipe
            mp_start = time.time()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            perf.log('mediapipe', (time.time() - mp_start) * 1000)
            
            face_detected = False
            left_result = None
            right_result = None
            drowsy_alert = False
            alarm_active = False
            yawn_state = None
            is_yawning = False
            head_result = None
            alarm_reason = ""
            
            if results.multi_face_landmarks:
                face_detected = True
                
                for face_lm in results.multi_face_landmarks:
                    landmarks = face_lm.landmark
                    
                    # Draw landmarks
                    draw_landmarks_custom(frame, landmarks, LEFT_EYE_INDICES, (0, 255, 0), 1)
                    draw_landmarks_custom(frame, landmarks, RIGHT_EYE_INDICES, (0, 255, 0), 1)
                    
                    yawn_color = (0, 0, 255) if is_yawning else (0, 255, 255)
                    draw_mouth_landmarks(frame, landmarks, yawn_color)
                    
                    # HEAD DROOP DETECTION (NEW)
                    hd_start = time.time()
                    head_result = head_detector.update(landmarks, w, h)
                    perf.log('head_droop', (time.time() - hd_start) * 1000)
                    
                    # Eye bounding boxes
                    left_bbox = get_eye_region(frame, landmarks, LEFT_EYE_INDICES)
                    right_bbox = get_eye_region(frame, landmarks, RIGHT_EYE_INDICES)
                    
                    # EAR
                    left_ear = calculate_ear(landmarks, LEFT_EYE_INDICES)
                    right_ear = calculate_ear(landmarks, RIGHT_EYE_INDICES)
                    
                    # Eye CNN
                    if eye_model is not None:
                        run_cnn = True
                        if SKIP_FRAME_MODE and frame_count % CNN_EVERY_N_FRAMES != 0:
                            run_cnn = False
                        
                        if run_cnn:
                            prep_start = time.time()
                            batch, valid = extract_both_eyes_batch(frame, left_bbox, right_bbox)
                            perf.log('preprocess', (time.time() - prep_start) * 1000)
                            
                            inf_start = time.time()
                            if valid[0] or valid[1]:
                                preds = eye_model.predict_batch(batch)
                                left_cnn = preds[0] if valid[0] else 0.5
                                right_cnn = preds[1] if valid[1] else 0.5
                                last_cnn = (left_cnn, right_cnn)
                            perf.log('inference', (time.time() - inf_start) * 1000)
                        else:
                            left_cnn, right_cnn = last_cnn
                        
                        # Eye fusion
                        fus_start = time.time()
                        head_pitch = head_result.pitch if head_result else None
                        left_result = left_eye_detector.update(left_cnn, left_ear, head_pitch=head_pitch)
                        right_result = right_eye_detector.update(right_cnn, right_ear, head_pitch=head_pitch)
                        perf.log('fusion', (time.time() - fus_start) * 1000)
                        
                        # Eye boxes
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
                        
                        # Drowsy counter
                        left_closed = left_result.state == EyeState.CLOSED
                        right_closed = right_result.state == EyeState.CLOSED
                        left_unc = left_result.state == EyeState.UNCERTAIN
                        right_unc = right_result.state == EyeState.UNCERTAIN
                        
                        if left_closed and right_closed:
                            both_closed_frames += 1
                        elif (left_closed and right_unc) or (right_closed and left_unc):
                            both_closed_frames += 0.75
                        elif left_unc and right_unc:
                            both_closed_frames += 0.5
                        else:
                            both_closed_frames = max(0, both_closed_frames - 2)
                    
                    # Yawn
                    yawn_start = time.time()
                    is_yawning, mar, yawn_state = yawn_detector.update(landmarks, h, w)
                    perf.log('yawn', (time.time() - yawn_start) * 1000)
                    recent_yawn_frames.append(1 if is_yawning else 0)
            
            # FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            
            # DROWSINESS DECISION
            if head_result and head_result.is_severely_drooping:
                # SEVERE HEAD DROOP - immediate alarm
                alarm_reason = "MICROSLEEP - HEAD DROOP"
                alarm.trigger_immediate(alarm_reason)
                drowsy_alert = True
                alarm_active = alarm.alarm_active
            else:
                # Check thresholds
                recent_yawn = sum(recent_yawn_frames) > 0 if recent_yawn_frames else False
                threshold = max(10, DROWSINESS_FRAME_THRESHOLD // 2) if recent_yawn else DROWSINESS_FRAME_THRESHOLD
                
                # Yawn fatigue
                yawn_count = yawn_state.get('yawn_count', 0) if yawn_state else 0
                yawn_freq = 0.0
                if yawn_state and yawn_detector.yawn_timestamps:
                    recent = [t for t in yawn_detector.yawn_timestamps if curr_time - t <= YAWN_OBSERVATION_WINDOW]
                    if len(recent) >= 2:
                        span = recent[-1] - recent[0]
                        if span >= 30:
                            yawn_freq = len(recent) / (span / 60.0)
                
                extreme_yawn = (yawn_count >= YAWN_MIN_COUNT_FOR_ALARM and yawn_freq >= YAWN_DANGER_THRESHOLD)
                
                # Head droop + eyes closed
                head_droop_eyes = False
                if head_result and head_result.is_drooping:
                    if left_result and right_result:
                        if left_result.state == EyeState.CLOSED or right_result.state == EyeState.CLOSED:
                            head_droop_eyes = True
                            alarm_reason = "HEAD DROOP + EYES CLOSED"
                
                # Combine
                eye_drowsy = both_closed_frames >= threshold
                
                if eye_drowsy:
                    alarm_reason = "EYES CLOSED"
                elif extreme_yawn:
                    alarm_reason = "EXCESSIVE YAWNING"
                
                drowsy_alert = eye_drowsy or extreme_yawn or head_droop_eyes
                alarm_active = alarm.update_state(drowsy_alert, alarm_reason)
            
            # Display
            disp_start = time.time()
            display_info_v53(frame, fps, face_detected, left_result, right_result,
                            int(both_closed_frames), drowsy_alert, alarm_active,
                            yawn_state, head_result,
                            perf if show_perf else None,
                            alarm.alarm_reason if alarm_active else "")
            perf.log('display', (time.time() - disp_start) * 1000)
            perf.log('total', (time.time() - frame_start) * 1000)
            
            # Console log
            console.log(fps, head_result, left_result, right_result, yawn_state,
                       int(both_closed_frames), alarm_active,
                       alarm.alarm_reason if alarm_active else "", perf if show_perf else None)
            
            cv2.imshow('Drowsiness Detection v5.3 (Geometric)', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[STOPPING] Terminated by user")
                break
            elif key == ord('p'):
                show_perf = not show_perf
                print(f"[INFO] Performance: {'ON' if show_perf else 'OFF'}")
            elif key == ord('d'):
                SHOW_HEAD_DEBUG = not SHOW_HEAD_DEBUG
                print(f"[INFO] Debug: {'ON' if SHOW_HEAD_DEBUG else 'OFF'}")
            elif key == ord('l'):
                ENABLE_CONSOLE_LOGGING = not ENABLE_CONSOLE_LOGGING
                print(f"[INFO] Console log: {'ON' if ENABLE_CONSOLE_LOGGING else 'OFF'}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    alarm.cleanup()
    
    # Summary
    print("\n" + "=" * 70)
    print("SESSION SUMMARY")
    print("=" * 70)
    summary = perf.get_summary()
    print(f"Timings: MP:{summary['mediapipe']:.1f} CNN:{summary['inference']:.1f} HD:{summary['head_droop']:.1f}ms")
    print(f"Total: {summary['total']:.1f}ms ({1000/max(0.001, summary['total']):.1f} FPS)")
    
    hd_stats = head_detector.get_stats()
    print(f"\nHead Droop: {hd_stats['avg_time_ms']:.2f}ms, droop_frames={hd_stats['droop_frames']}")
    print(f"Yawns: {yawn_detector.yawn_count}")
    print("=" * 70)


if __name__ == "__main__":
    main()
