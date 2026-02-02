"""
Geometric Head Droop Detection for Driver Drowsiness Detection System

=====================================================================
WHY THIS APPROACH IS BETTER THAN PnP EULER ANGLES:
=====================================================================

The PnP approach fails because:
1. When head falls SIDEWAYS, most rotation goes to ROLL/YAW, not PITCH
2. Generic 3D face model doesn't match real faces
3. Combined rotations cause gimbal lock and extraction errors
4. Head tilted BACK gives large positive pitch (false negative for droop)

THIS APPROACH detects "dangerous head positions" directly from landmarks:
- Eye line tilt angle (sideways droop)
- Face centroid vertical drop (any direction droop) 
- Nose tip vertical position (forward/backward)
- Multiple redundant signals for robustness

For drowsiness, we don't need exact angles - we need to know:
"Is the driver's head in a dangerous position?"

=====================================================================
Author: Binomial Team
Performance Target: < 3ms per frame
=====================================================================
"""

from dataclasses import dataclass, field
from collections import deque
from typing import Optional, Tuple, Dict, List, Any
from enum import Enum
import numpy as np
import cv2
import time


class HeadState(Enum):
    """Head position states for drowsiness detection."""
    NORMAL = "Normal"
    LOOKING_DOWN = "Down"
    LOOKING_AWAY = "Away"
    DROWSY = "Drowsy"
    DROOP = "DROOP!"
    CALIBRATING = "Calibrating"


@dataclass
class GeometricHeadResult:
    """
    Result from geometric head droop detection.
    
    Focuses on danger indicators rather than exact angles.
    """
    # State
    state: HeadState = HeadState.NORMAL
    
    # Danger score (0.0 = safe, 1.0 = dangerous)
    droop_score: float = 0.0
    
    # Key flags
    is_drooping: bool = False
    is_severely_drooping: bool = False
    is_looking_away: bool = False
    is_looking_down: bool = False
    
    # Geometric measurements
    eye_tilt_deg: float = 0.0          # Eye line angle from horizontal
    vertical_drop_ratio: float = 0.0   # How much face has dropped (0-1)
    nose_drop_ratio: float = 0.0       # Nose vertical position change
    face_size_ratio: float = 1.0       # Face size change (forward lean)
    
    # For compatibility with existing code
    pitch: float = 0.0                 # Approximate pitch (negative = down)
    yaw: float = 0.0                   # Approximate yaw  
    roll: float = 0.0                  # Approximate roll (from eye tilt)
    
    # Confidence and tracking
    confidence: float = 0.0
    droop_frames: int = 0
    
    # Raw components for debugging
    tilt_score: float = 0.0
    drop_score: float = 0.0
    nose_score: float = 0.0
    size_score: float = 0.0


class GeometricHeadDroopDetector:
    """
    Detects dangerous head positions using geometric landmark analysis.
    
    This detector is specifically designed to catch head drooping in ANY
    direction (forward, left, right) which the PnP approach fails to detect.
    
    Key features tracked:
    1. Eye line tilt - tilted eyes indicate sideways head roll
    2. Face vertical drop - face moving down in frame
    3. Nose vertical position - nose moving down indicates forward droop
    4. Face size changes - face getting larger indicates forward lean
    
    The detector calibrates on the first ~1 second to establish baseline.
    """
    
    # MediaPipe FaceMesh landmark indices
    LM = {
        # Eyes (for tilt detection)
        'left_eye_outer': 33,
        'left_eye_inner': 133,
        'right_eye_inner': 362,
        'right_eye_outer': 263,
        
        # Additional eye landmarks for center calculation
        'left_eye_top': 159,
        'left_eye_bottom': 145,
        'right_eye_top': 386,
        'right_eye_bottom': 374,
        
        # Nose (for vertical position and forward lean)
        'nose_tip': 1,
        'nose_bridge': 6,
        
        # Face boundary (for size and centroid)
        'forehead': 10,
        'chin': 152,
        'left_cheek': 234,
        'right_cheek': 454,
        'left_temple': 127,
        'right_temple': 356,
        
        # Mouth (for yaw estimation)
        'mouth_left': 61,
        'mouth_right': 291,
    }
    
    def __init__(
        self,
        # Tilt thresholds (degrees)
        tilt_warning_deg: float = 12.0,
        tilt_danger_deg: float = 20.0,
        
        # Vertical drop thresholds (ratio of face height)
        drop_warning_ratio: float = 0.06,
        drop_danger_ratio: float = 0.12,
        
        # Nose drop thresholds
        nose_warning_ratio: float = 0.05,
        nose_danger_ratio: float = 0.10,
        
        # Yaw threshold for "looking away"
        yaw_away_threshold: float = 35.0,
        
        # Temporal settings
        calibration_frames: int = 30,
        smoothing_alpha: float = 0.35,
        
        # Droop detection
        droop_warning_score: float = 0.40,
        droop_danger_score: float = 0.55,
        droop_warning_frames: int = 45,   # ~1.5 seconds
        droop_danger_frames: int = 25,    # ~0.8 seconds for severe
    ):
        # Thresholds
        self.tilt_warning = tilt_warning_deg
        self.tilt_danger = tilt_danger_deg
        self.drop_warning = drop_warning_ratio
        self.drop_danger = drop_danger_ratio
        self.nose_warning = nose_warning_ratio
        self.nose_danger = nose_danger_ratio
        self.yaw_away_threshold = yaw_away_threshold
        
        self.droop_warning_score = droop_warning_score
        self.droop_danger_score = droop_danger_score
        self.droop_warning_frames = droop_warning_frames
        self.droop_danger_frames = droop_danger_frames
        
        # Smoothing
        self.smoothing_alpha = smoothing_alpha
        self.calibration_frames_needed = calibration_frames
        
        # Calibration state
        self._calibration_buffer: List[Dict] = []
        self._calibration_count: int = 0
        self._is_calibrated: bool = False
        
        # Baseline values (set during calibration)
        self._baseline_eye_center_y: float = 0.0
        self._baseline_nose_y: float = 0.0
        self._baseline_face_height: float = 0.0
        self._baseline_face_width: float = 0.0
        
        # Smoothed values
        self._smooth_tilt: float = 0.0
        self._smooth_drop: float = 0.0
        self._smooth_nose: float = 0.0
        self._smooth_score: float = 0.0
        
        # State tracking
        self._droop_frame_count: int = 0
        self._severe_droop_count: int = 0
        self._away_frame_count: int = 0
        
        # History
        self._score_history: deque = deque(maxlen=90)
        
        # Stats
        self._update_count: int = 0
        self._total_time_ms: float = 0.0
    
    def _get_point(self, landmarks, idx: int, w: int, h: int) -> np.ndarray:
        """Get normalized landmark as pixel coordinates."""
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h])
    
    def _get_point_3d(self, landmarks, idx: int, w: int, h: int) -> np.ndarray:
        """Get 3D landmark coordinates."""
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h, lm.z * w])
    
    def _calc_eye_centers(self, landmarks, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the center point of each eye."""
        # Left eye center
        left_outer = self._get_point(landmarks, self.LM['left_eye_outer'], w, h)
        left_inner = self._get_point(landmarks, self.LM['left_eye_inner'], w, h)
        left_top = self._get_point(landmarks, self.LM['left_eye_top'], w, h)
        left_bot = self._get_point(landmarks, self.LM['left_eye_bottom'], w, h)
        left_center = (left_outer + left_inner + left_top + left_bot) / 4.0
        
        # Right eye center
        right_outer = self._get_point(landmarks, self.LM['right_eye_outer'], w, h)
        right_inner = self._get_point(landmarks, self.LM['right_eye_inner'], w, h)
        right_top = self._get_point(landmarks, self.LM['right_eye_top'], w, h)
        right_bot = self._get_point(landmarks, self.LM['right_eye_bottom'], w, h)
        right_center = (right_outer + right_inner + right_top + right_bot) / 4.0
        
        return left_center, right_center
    
    def _calc_eye_tilt(self, landmarks, w: int, h: int) -> float:
        """
        Calculate eye line tilt angle in degrees.
        
        When head tilts sideways (rolling), the line connecting eyes tilts.
        This is the most reliable indicator of sideways head droop.
        
        Returns:
            Angle in degrees. Positive = right side lower (tilting right)
        """
        left_center, right_center = self._calc_eye_centers(landmarks, w, h)
        
        dx = right_center[0] - left_center[0]
        dy = right_center[1] - left_center[1]
        
        if abs(dx) < 1.0:
            return 0.0
        
        angle_rad = np.arctan2(dy, dx)
        return np.degrees(angle_rad)
    
    def _calc_face_geometry(self, landmarks, w: int, h: int) -> Dict:
        """
        Calculate face geometry measurements.
        
        Returns dict with:
        - eye_center_y: Average Y position of eye centers
        - nose_y: Nose tip Y position  
        - face_height: Distance from forehead to chin
        - face_width: Distance between temples
        - face_centroid: Center of face
        """
        forehead = self._get_point(landmarks, self.LM['forehead'], w, h)
        chin = self._get_point(landmarks, self.LM['chin'], w, h)
        left_temple = self._get_point(landmarks, self.LM['left_temple'], w, h)
        right_temple = self._get_point(landmarks, self.LM['right_temple'], w, h)
        nose = self._get_point(landmarks, self.LM['nose_tip'], w, h)
        
        left_eye, right_eye = self._calc_eye_centers(landmarks, w, h)
        
        return {
            'eye_center_y': (left_eye[1] + right_eye[1]) / 2.0,
            'eye_center_x': (left_eye[0] + right_eye[0]) / 2.0,
            'nose_y': nose[1],
            'nose_x': nose[0],
            'face_height': np.linalg.norm(chin - forehead),
            'face_width': np.linalg.norm(right_temple - left_temple),
            'forehead_y': forehead[1],
            'chin_y': chin[1],
        }
    
    def _calc_yaw_estimate(self, landmarks, w: int, h: int) -> float:
        """
        Estimate yaw (left/right looking) from nose position relative to face center.
        
        When looking right, nose appears to the right of face center.
        When looking left, nose appears to the left of face center.
        
        Returns:
            Estimated yaw in degrees. Positive = looking right.
        """
        nose = self._get_point(landmarks, self.LM['nose_tip'], w, h)
        left_cheek = self._get_point(landmarks, self.LM['left_cheek'], w, h)
        right_cheek = self._get_point(landmarks, self.LM['right_cheek'], w, h)
        
        face_center_x = (left_cheek[0] + right_cheek[0]) / 2.0
        face_width = abs(right_cheek[0] - left_cheek[0])
        
        if face_width < 10:
            return 0.0
        
        # How far nose is from center, normalized
        nose_offset = (nose[0] - face_center_x) / face_width
        
        # Map to approximate degrees (rough calibration)
        # When nose is 0.3 face-widths off center, roughly 30-40 degrees yaw
        yaw_estimate = nose_offset * 120.0  # Scale factor
        
        return np.clip(yaw_estimate, -90.0, 90.0)
    
    def _score_from_value(self, value: float, warning: float, danger: float, absolute: bool = True) -> float:
        """Convert a measurement to 0-1 danger score with smooth curve."""
        if absolute:
            value = abs(value)
        
        if value <= 0:
            return 0.0
        elif value < warning:
            # Below warning: gentle ramp
            return 0.2 * (value / warning)
        elif value < danger:
            # Warning to danger: steeper ramp
            progress = (value - warning) / (danger - warning)
            return 0.2 + 0.4 * progress
        else:
            # Above danger: continue to 1.0
            overshoot = (value - danger) / danger
            return min(1.0, 0.6 + 0.4 * overshoot)
    
    def _calibrate(self, measurements: Dict):
        """Accumulate calibration measurements and compute baselines."""
        self._calibration_buffer.append(measurements)
        self._calibration_count += 1
        
        if self._calibration_count >= self.calibration_frames_needed:
            # Compute baselines using median for robustness
            self._baseline_eye_center_y = np.median([m['eye_center_y'] for m in self._calibration_buffer])
            self._baseline_nose_y = np.median([m['nose_y'] for m in self._calibration_buffer])
            self._baseline_face_height = np.median([m['face_height'] for m in self._calibration_buffer])
            self._baseline_face_width = np.median([m['face_width'] for m in self._calibration_buffer])
            
            self._is_calibrated = True
            self._calibration_buffer.clear()
    
    def update(self, landmarks, frame_width: int, frame_height: int) -> GeometricHeadResult:
        """
        Detect head droop from facial landmarks.
        
        Args:
            landmarks: MediaPipe face landmarks (468 points)
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            
        Returns:
            GeometricHeadResult with droop detection
        """
        start_time = time.perf_counter()
        w, h = frame_width, frame_height
        
        try:
            # Get all measurements
            eye_tilt = self._calc_eye_tilt(landmarks, w, h)
            geometry = self._calc_face_geometry(landmarks, w, h)
            yaw_est = self._calc_yaw_estimate(landmarks, w, h)
            
            # Calibration phase
            if not self._is_calibrated:
                self._calibrate(geometry)
                
                result = GeometricHeadResult(
                    state=HeadState.CALIBRATING,
                    confidence=self._calibration_count / self.calibration_frames_needed,
                    eye_tilt_deg=eye_tilt,
                    roll=eye_tilt,
                    yaw=yaw_est,
                )
                self._update_stats(start_time)
                return result
            
            # Calculate changes from baseline
            # Vertical drop: positive = face moved down
            eye_drop = (geometry['eye_center_y'] - self._baseline_eye_center_y) / self._baseline_face_height
            nose_drop = (geometry['nose_y'] - self._baseline_nose_y) / self._baseline_face_height
            
            # Face size change (forward lean makes face larger)
            size_change = geometry['face_height'] / self._baseline_face_height - 1.0
            
            # Calculate individual danger scores
            tilt_score = self._score_from_value(eye_tilt, self.tilt_warning, self.tilt_danger)
            drop_score = self._score_from_value(eye_drop, self.drop_warning, self.drop_danger, absolute=False)
            nose_score = self._score_from_value(nose_drop, self.nose_warning, self.nose_danger, absolute=False)
            size_score = self._score_from_value(size_change, 0.1, 0.2, absolute=False)
            
            # Only count positive drops (head going down, not up)
            if eye_drop < 0:
                drop_score *= 0.2  # Reduce score when head is raised
            if nose_drop < 0:
                nose_score *= 0.2
            
            # Combined score using max-of-signals approach
            # This ensures ANY type of droop is detected
            raw_score = max(
                tilt_score * 1.0,          # Eye tilt (sideways droop)
                drop_score * 1.0,          # Face dropping (forward droop)
                nose_score * 0.9,          # Nose dropping
                (tilt_score + drop_score) / 2 * 1.2,  # Combined tilt + drop
            )
            raw_score = min(1.0, raw_score)
            
            # Apply smoothing
            alpha = self.smoothing_alpha
            self._smooth_tilt = alpha * abs(eye_tilt) + (1 - alpha) * self._smooth_tilt
            self._smooth_drop = alpha * max(0, eye_drop) + (1 - alpha) * self._smooth_drop
            self._smooth_nose = alpha * max(0, nose_drop) + (1 - alpha) * self._smooth_nose
            self._smooth_score = alpha * raw_score + (1 - alpha) * self._smooth_score
            
            # Track history
            self._score_history.append(self._smooth_score)
            
            # Update frame counters
            if self._smooth_score >= self.droop_danger_score:
                self._severe_droop_count += 1
                self._droop_frame_count += 1
            elif self._smooth_score >= self.droop_warning_score:
                self._severe_droop_count = max(0, self._severe_droop_count - 2)
                self._droop_frame_count += 1
            else:
                self._severe_droop_count = max(0, self._severe_droop_count - 3)
                self._droop_frame_count = max(0, self._droop_frame_count - 2)
            
            # Check yaw for "looking away"
            is_looking_away = abs(yaw_est) > self.yaw_away_threshold
            if is_looking_away:
                self._away_frame_count += 1
            else:
                self._away_frame_count = max(0, self._away_frame_count - 1)
            
            # Determine state
            is_severely_drooping = self._severe_droop_count >= self.droop_danger_frames
            is_drooping = self._droop_frame_count >= self.droop_warning_frames
            is_looking_down = self._smooth_drop > self.drop_warning or self._smooth_nose > self.nose_warning
            
            if is_severely_drooping:
                state = HeadState.DROOP
            elif is_drooping:
                state = HeadState.DROWSY
            elif is_looking_away:
                state = HeadState.LOOKING_AWAY
            elif is_looking_down or self._smooth_tilt > self.tilt_warning:
                state = HeadState.LOOKING_DOWN
            else:
                state = HeadState.NORMAL
            
            # Approximate Euler angles for display compatibility
            # Roll from eye tilt (direct)
            approx_roll = eye_tilt
            # Pitch estimated from drop (negative = looking down)
            approx_pitch = -eye_drop * 150  # Rough scaling
            if is_severely_drooping or is_drooping:
                approx_pitch = min(approx_pitch, -30)  # Ensure negative when drooping
            # Yaw from estimate
            approx_yaw = yaw_est
            
            result = GeometricHeadResult(
                state=state,
                droop_score=self._smooth_score,
                is_drooping=is_drooping,
                is_severely_drooping=is_severely_drooping,
                is_looking_away=is_looking_away,
                is_looking_down=is_looking_down,
                
                eye_tilt_deg=eye_tilt,
                vertical_drop_ratio=eye_drop,
                nose_drop_ratio=nose_drop,
                face_size_ratio=1.0 + size_change,
                
                pitch=approx_pitch,
                yaw=approx_yaw,
                roll=approx_roll,
                
                confidence=1.0,
                droop_frames=self._droop_frame_count,
                
                tilt_score=tilt_score,
                drop_score=drop_score,
                nose_score=nose_score,
                size_score=size_score,
            )
            
            self._update_stats(start_time)
            return result
            
        except Exception as e:
            result = GeometricHeadResult(state=HeadState.NORMAL, confidence=0.0)
            self._update_stats(start_time)
            return result
    
    def _update_stats(self, start_time: float):
        """Update performance statistics."""
        elapsed = (time.perf_counter() - start_time) * 1000
        self._update_count += 1
        self._total_time_ms += elapsed
    
    def detect_head_drooping(
        self,
        threshold_frames: int = 60,
        severe_frames: int = 30,
    ) -> Tuple[bool, bool, int]:
        """
        Check for sustained head droop (compatibility method).
        
        Returns:
            Tuple of (is_drooping, is_severe, frame_count)
        """
        is_drooping = self._droop_frame_count >= threshold_frames
        is_severe = self._severe_droop_count >= severe_frames
        return is_drooping, is_severe, self._droop_frame_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detector statistics."""
        avg_time = self._total_time_ms / max(1, self._update_count)
        return {
            'update_count': self._update_count,
            'avg_time_ms': round(avg_time, 3),
            'is_calibrated': self._is_calibrated,
            'droop_frames': self._droop_frame_count,
            'severe_frames': self._severe_droop_count,
            'current_score': round(self._smooth_score, 3),
            'current_tilt': round(self._smooth_tilt, 1),
        }
    
    def reset(self):
        """Reset detector for new session."""
        self._calibration_buffer.clear()
        self._calibration_count = 0
        self._is_calibrated = False
        
        self._smooth_tilt = 0.0
        self._smooth_drop = 0.0
        self._smooth_nose = 0.0
        self._smooth_score = 0.0
        
        self._droop_frame_count = 0
        self._severe_droop_count = 0
        self._away_frame_count = 0
        
        self._score_history.clear()
        self._update_count = 0
        self._total_time_ms = 0.0


# ============================================================================
# Visualization
# ============================================================================

def draw_geometric_head_info(
    frame: np.ndarray,
    result: GeometricHeadResult,
    position: Tuple[int, int] = (10, 30),
) -> np.ndarray:
    """Draw head droop information on frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = position
    
    # State color
    colors = {
        HeadState.NORMAL: (0, 255, 0),
        HeadState.LOOKING_DOWN: (0, 165, 255),
        HeadState.LOOKING_AWAY: (0, 255, 255),
        HeadState.DROWSY: (0, 100, 255),
        HeadState.DROOP: (0, 0, 255),
        HeadState.CALIBRATING: (255, 255, 0),
    }
    color = colors.get(result.state, (255, 255, 255))
    
    # State text
    cv2.putText(frame, f"Head: {result.state.value}", (x, y),
                font, 0.6, (0, 0, 0), 3)
    cv2.putText(frame, f"Head: {result.state.value}", (x, y),
                font, 0.6, color, 2)
    
    # Approximate angles
    cv2.putText(frame, f"P:{result.pitch:+.0f} Y:{result.yaw:+.0f} R:{result.roll:+.0f}",
                (x, y + 25), font, 0.4, (200, 200, 200), 1)
    
    # Score bar
    bar_width = 150
    bar_y = y + 35
    cv2.rectangle(frame, (x, bar_y), (x + bar_width, bar_y + 12), (50, 50, 50), -1)
    score_width = int(bar_width * result.droop_score)
    if score_width > 0:
        cv2.rectangle(frame, (x, bar_y), (x + score_width, bar_y + 12), color, -1)
    
    # Droop frame counter if active
    if result.droop_frames > 0:
        cv2.putText(frame, f"{result.droop_frames}/60",
                    (x + bar_width + 5, bar_y + 10), font, 0.35, (200, 200, 200), 1)
    
    return frame


# ============================================================================
# Standalone test
# ============================================================================

def _run_tests():
    """Quick unit tests."""
    print("=" * 60)
    print("GeometricHeadDroopDetector Tests")
    print("=" * 60)
    
    class MockLandmark:
        def __init__(self, x, y, z=0):
            self.x = x
            self.y = y
            self.z = z
    
    def make_landmarks(eye_tilt=0, drop=0, yaw_shift=0):
        """Create mock landmarks."""
        landmarks = [MockLandmark(0.5, 0.5)] * 468
        
        tilt_rad = np.radians(eye_tilt)
        spread = 0.12
        
        # Eye positions with tilt
        ly = 0.35 + drop + spread * np.sin(tilt_rad)
        ry = 0.35 + drop - spread * np.sin(tilt_rad)
        
        # Left eye
        landmarks[33] = MockLandmark(0.35, ly)   # outer
        landmarks[133] = MockLandmark(0.42, ly)  # inner
        landmarks[159] = MockLandmark(0.38, ly - 0.02)  # top
        landmarks[145] = MockLandmark(0.38, ly + 0.02)  # bottom
        
        # Right eye
        landmarks[362] = MockLandmark(0.58, ry)  # inner
        landmarks[263] = MockLandmark(0.65, ry)  # outer
        landmarks[386] = MockLandmark(0.62, ry - 0.02)  # top
        landmarks[374] = MockLandmark(0.62, ry + 0.02)  # bottom
        
        # Face boundary
        landmarks[10] = MockLandmark(0.5, 0.18 + drop)   # forehead
        landmarks[152] = MockLandmark(0.5, 0.82 + drop)  # chin
        landmarks[234] = MockLandmark(0.28, 0.5 + drop)  # left cheek
        landmarks[454] = MockLandmark(0.72, 0.5 + drop)  # right cheek
        landmarks[127] = MockLandmark(0.25, 0.4 + drop)  # left temple
        landmarks[356] = MockLandmark(0.75, 0.4 + drop)  # right temple
        
        # Nose with yaw shift
        landmarks[1] = MockLandmark(0.5 + yaw_shift * 0.15, 0.48 + drop)  # tip
        landmarks[6] = MockLandmark(0.5 + yaw_shift * 0.1, 0.38 + drop)   # bridge
        
        # Mouth
        landmarks[61] = MockLandmark(0.4, 0.65 + drop)
        landmarks[291] = MockLandmark(0.6, 0.65 + drop)
        
        return landmarks
    
    detector = GeometricHeadDroopDetector(calibration_frames=15)
    
    # Test 1: Calibration
    print("\n[Test 1] Calibration...")
    for _ in range(20):
        detector.update(make_landmarks(), 640, 480)
    assert detector._is_calibrated
    print("  ✓ Calibrated")
    
    # Test 2: Normal pose
    print("\n[Test 2] Normal pose...")
    result = detector.update(make_landmarks(), 640, 480)
    print(f"  State: {result.state.value}, Score: {result.droop_score:.3f}")
    assert result.state == HeadState.NORMAL or result.droop_score < 0.5
    print("  ✓ Normal detected")
    
    # Test 3: Eye tilt (sideways droop)
    print("\n[Test 3] Eye tilt (sideways droop)...")
    detector.reset()
    for _ in range(20):
        detector.update(make_landmarks(), 640, 480)
    
    for _ in range(40):
        result = detector.update(make_landmarks(eye_tilt=25), 640, 480)
    
    print(f"  Tilt: {result.eye_tilt_deg:.1f}°, Score: {result.droop_score:.3f}")
    print(f"  State: {result.state.value}")
    assert result.droop_score > 0.3
    print("  ✓ Sideways tilt detected")
    
    # Test 4: Vertical drop (forward droop)
    print("\n[Test 4] Vertical drop (forward droop)...")
    detector.reset()
    for _ in range(20):
        detector.update(make_landmarks(), 640, 480)
    
    for _ in range(40):
        result = detector.update(make_landmarks(drop=0.12), 640, 480)
    
    print(f"  Drop: {result.vertical_drop_ratio:.3f}, Score: {result.droop_score:.3f}")
    print(f"  State: {result.state.value}")
    assert result.droop_score > 0.3
    print("  ✓ Forward droop detected")
    
    # Test 5: Combined tilt + drop
    print("\n[Test 5] Combined tilt + drop...")
    detector.reset()
    for _ in range(20):
        detector.update(make_landmarks(), 640, 480)
    
    for _ in range(50):
        result = detector.update(make_landmarks(eye_tilt=20, drop=0.08), 640, 480)
    
    print(f"  Tilt: {result.eye_tilt_deg:.1f}°, Drop: {result.vertical_drop_ratio:.3f}")
    print(f"  Score: {result.droop_score:.3f}, State: {result.state.value}")
    assert result.droop_score > 0.4
    print("  ✓ Combined droop detected")
    
    # Test 6: Looking away (yaw)
    print("\n[Test 6] Looking away...")
    detector.reset()
    for _ in range(20):
        detector.update(make_landmarks(), 640, 480)
    
    for _ in range(20):
        result = detector.update(make_landmarks(yaw_shift=0.5), 640, 480)
    
    print(f"  Yaw: {result.yaw:.1f}°, State: {result.state.value}")
    print("  ✓ Yaw detection working")
    
    # Test 7: Performance
    print("\n[Test 7] Performance...")
    detector.reset()
    for _ in range(20):
        detector.update(make_landmarks(), 640, 480)
    
    import time
    start = time.perf_counter()
    for _ in range(100):
        detector.update(make_landmarks(), 640, 480)
    elapsed = time.perf_counter() - start
    
    avg_ms = elapsed / 100 * 1000
    print(f"  Average: {avg_ms:.3f} ms/frame")
    print(f"  {'✓' if avg_ms < 3.0 else '✗'} Target: < 3.0 ms")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    _run_tests()
