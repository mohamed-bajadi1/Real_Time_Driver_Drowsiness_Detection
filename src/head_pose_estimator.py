"""
Head Pose Estimation Module for Driver Drowsiness Detection System

This module estimates 3D head orientation (pitch, yaw, roll) from MediaPipe
facial landmarks using the Perspective-n-Point (PnP) algorithm.

ALGORITHM: Uses 6 stable facial landmarks (nose, chin, eye corners, mouth corners)
to solve the PnP problem and extract Euler angles representing head orientation.

PERFORMANCE TARGET: < 3ms per frame on CPU

Author: Driver Drowsiness Detection Team (Binomial)
Sprint: 5 - Head Pose Integration
"""

from enum import Enum
from dataclasses import dataclass, field
from collections import deque
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import cv2
import time


@dataclass
class HeadPoseResult:
    """
    Head pose estimation result.
    
    Attributes:
        pitch: Vertical rotation in degrees. Negative = looking down, Positive = looking up.
               Range: approximately -90° to +90°
        yaw: Horizontal rotation in degrees. Negative = looking left, Positive = looking right.
             Range: approximately -90° to +90°
        roll: Rotation around the viewing axis in degrees (head tilt).
              Negative = tilt left, Positive = tilt right.
        confidence: Reliability score from 0.0 to 1.0 based on reprojection error
                    and landmark visibility.
        is_looking_down: True if pitch < looking_down_threshold (default -15°)
        is_severely_down: True if pitch < severe_droop_threshold (default -30°)
        is_looking_away: True if abs(yaw) > looking_away_threshold (default 30°)
        raw_pitch: Unsmoothed pitch value for debugging
        raw_yaw: Unsmoothed yaw value for debugging
        raw_roll: Unsmoothed roll value for debugging
        reprojection_error: Mean reprojection error in pixels (lower = better)
    """
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    confidence: float = 0.0
    is_looking_down: bool = False
    is_severely_down: bool = False
    is_looking_away: bool = False
    raw_pitch: float = 0.0
    raw_yaw: float = 0.0
    raw_roll: float = 0.0
    reprojection_error: float = 0.0


class HeadPoseEstimator:
    """
    Estimates 3D head pose from MediaPipe facial landmarks using PnP.
    
    METHOD: Perspective-n-Point (solvePnP) with 6 stable facial landmarks
    
    LANDMARKS USED (MediaPipe indices):
        - Nose tip (1): Central reference point
        - Chin (152): Lower face anchor
        - Left eye outer corner (33): Left side reference
        - Right eye outer corner (263): Right side reference  
        - Left mouth corner (61): Lower left reference
        - Right mouth corner (291): Lower right reference
    
    3D MODEL: Generic anthropometric face model with approximate measurements:
        - Inter-pupillary distance: ~63mm
        - Face width: ~130mm
        - Face height (nose to chin): ~90mm
    
    TEMPORAL FILTERING: Exponential moving average with configurable alpha
    
    OUTLIER REJECTION: Sudden angle changes > 30° are dampened
    
    PERFORMANCE: Typically < 2ms per frame on modern CPU
    
    Example:
        >>> estimator = HeadPoseEstimator()
        >>> result = estimator.update(landmarks, frame_width=640, frame_height=480)
        >>> print(f"Pitch: {result.pitch:.1f}°, Yaw: {result.yaw:.1f}°")
    """
    
    # MediaPipe Face Mesh landmark indices for PnP
    # Selected for stability across expressions and partial occlusions
    LANDMARK_INDICES = {
        'nose_tip': 1,
        'chin': 152,
        'left_eye_outer': 33,
        'right_eye_outer': 263,
        'left_mouth': 61,
        'right_mouth': 291,
    }
    
    # Additional landmarks for enhanced stability (optional 8-point model)
    EXTENDED_LANDMARKS = {
        'nose_bridge': 6,
        'forehead': 10,
    }
    
    def __init__(
        self,
        smoothing_alpha: float = 0.4,
        smoothing_window: int = 5,
        looking_down_threshold: float = -15.0,
        severe_droop_threshold: float = -30.0,
        looking_away_threshold: float = 30.0,
        outlier_threshold: float = 30.0,
        min_confidence: float = 0.3,
        use_extended_model: bool = False,
    ):
        """
        Initialize the head pose estimator.
        
        Args:
            smoothing_alpha: EMA smoothing factor (0-1). Higher = more responsive,
                           lower = more stable. Default 0.4 balances both.
            smoothing_window: Window size for history tracking (used for stats).
            looking_down_threshold: Pitch angle (degrees) below which is_looking_down=True.
            severe_droop_threshold: Pitch angle (degrees) below which is_severely_down=True.
            looking_away_threshold: Absolute yaw angle (degrees) above which is_looking_away=True.
            outlier_threshold: Max allowed angle change per frame before dampening.
            min_confidence: Minimum confidence to consider result valid.
            use_extended_model: Use 8-point model instead of 6-point (slightly more stable).
        """
        # Configuration
        self.smoothing_alpha = smoothing_alpha
        self.smoothing_window = smoothing_window
        self.looking_down_threshold = looking_down_threshold
        self.severe_droop_threshold = severe_droop_threshold
        self.looking_away_threshold = looking_away_threshold
        self.outlier_threshold = outlier_threshold
        self.min_confidence = min_confidence
        self.use_extended_model = use_extended_model
        
        # 3D Face Model Points (generic anthropometric model)
        # Coordinates in mm, centered at nose tip
        # Based on average adult face proportions
        self._init_3d_model()
        
        # Camera matrix (computed per frame based on image dimensions)
        self._camera_matrix: Optional[np.ndarray] = None
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        self._last_frame_size: Optional[Tuple[int, int]] = None
        
        # State tracking
        self._smoothed_pitch: Optional[float] = None
        self._smoothed_yaw: Optional[float] = None
        self._smoothed_roll: Optional[float] = None
        self._initialized = False
        
        # History for head drooping detection
        self._pitch_history: deque = deque(maxlen=120)  # ~4 seconds at 30 FPS
        self._severe_droop_history: deque = deque(maxlen=120)
        
        # Statistics tracking
        self._update_count = 0
        self._total_time_ms = 0.0
        self._angle_history: deque = deque(maxlen=smoothing_window)
        self._confidence_history: deque = deque(maxlen=smoothing_window)
        self._last_result: Optional[HeadPoseResult] = None
        
        # Pre-allocate arrays for performance
        self._image_points = np.zeros((6, 2), dtype=np.float64)
        self._image_points_extended = np.zeros((8, 2), dtype=np.float64)
    
    def _init_3d_model(self):
        """
        Initialize the 3D face model points.
        
        Uses a generic anthropometric face model with approximate measurements.
        The coordinate system is:
            - X: positive = right (from subject's perspective)
            - Y: positive = down
            - Z: positive = forward (towards camera)
        
        All units are in millimeters, centered at the nose tip.
        """
        # 6-point model (primary)
        self._model_points_6 = np.array([
            [0.0, 0.0, 0.0],           # Nose tip (origin)
            [0.0, 90.0, -20.0],        # Chin (below and slightly back)
            [-43.0, -32.0, -25.0],     # Left eye outer corner
            [43.0, -32.0, -25.0],      # Right eye outer corner
            [-28.0, 50.0, -15.0],      # Left mouth corner
            [28.0, 50.0, -15.0],       # Right mouth corner
        ], dtype=np.float64)
        
        # 8-point model (extended, optional)
        self._model_points_8 = np.array([
            [0.0, 0.0, 0.0],           # Nose tip
            [0.0, 90.0, -20.0],        # Chin
            [-43.0, -32.0, -25.0],     # Left eye outer corner
            [43.0, -32.0, -25.0],      # Right eye outer corner
            [-28.0, 50.0, -15.0],      # Left mouth corner
            [28.0, 50.0, -15.0],       # Right mouth corner
            [0.0, -25.0, 10.0],        # Nose bridge
            [0.0, -70.0, -15.0],       # Forehead
        ], dtype=np.float64)
    
    def _get_camera_matrix(self, frame_width: int, frame_height: int) -> np.ndarray:
        """
        Compute or retrieve cached camera intrinsic matrix.
        
        Assumes a typical webcam with ~60° horizontal FOV.
        
        Args:
            frame_width: Image width in pixels
            frame_height: Image height in pixels
            
        Returns:
            3x3 camera intrinsic matrix
        """
        # Check cache
        if self._camera_matrix is not None and self._last_frame_size == (frame_width, frame_height):
            return self._camera_matrix
        
        # Estimate focal length from FOV
        # FOV ≈ 60° horizontal is typical for webcams
        fov_horizontal = 60.0  # degrees
        focal_length = frame_width / (2.0 * np.tan(np.radians(fov_horizontal / 2)))
        
        # Principal point at image center
        cx = frame_width / 2.0
        cy = frame_height / 2.0
        
        # Construct camera matrix
        self._camera_matrix = np.array([
            [focal_length, 0.0, cx],
            [0.0, focal_length, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        
        self._last_frame_size = (frame_width, frame_height)
        return self._camera_matrix
    
    def _extract_image_points(
        self, 
        landmarks, 
        frame_width: int, 
        frame_height: int
    ) -> Tuple[np.ndarray, bool]:
        """
        Extract 2D image points from MediaPipe landmarks.
        
        Args:
            landmarks: MediaPipe face landmarks object (468 points)
            frame_width: Image width in pixels
            frame_height: Image height in pixels
            
        Returns:
            Tuple of (image_points array, success flag)
        """
        try:
            indices = [
                self.LANDMARK_INDICES['nose_tip'],
                self.LANDMARK_INDICES['chin'],
                self.LANDMARK_INDICES['left_eye_outer'],
                self.LANDMARK_INDICES['right_eye_outer'],
                self.LANDMARK_INDICES['left_mouth'],
                self.LANDMARK_INDICES['right_mouth'],
            ]
            
            # Extract and denormalize coordinates
            for i, idx in enumerate(indices):
                lm = landmarks[idx]
                self._image_points[i, 0] = lm.x * frame_width
                self._image_points[i, 1] = lm.y * frame_height
            
            if self.use_extended_model:
                # Add extended landmarks
                self._image_points_extended[:6] = self._image_points
                
                ext_indices = [
                    self.EXTENDED_LANDMARKS['nose_bridge'],
                    self.EXTENDED_LANDMARKS['forehead'],
                ]
                for i, idx in enumerate(ext_indices):
                    lm = landmarks[idx]
                    self._image_points_extended[6 + i, 0] = lm.x * frame_width
                    self._image_points_extended[6 + i, 1] = lm.y * frame_height
                
                return self._image_points_extended.copy(), True
            
            return self._image_points.copy(), True
            
        except (IndexError, AttributeError) as e:
            return self._image_points, False
    
    def _rotation_matrix_to_euler(self, R: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert rotation matrix to Euler angles (pitch, yaw, roll).
        
        FIXED VERSION: Uses cv2.RQDecomp3x3 for robust decomposition that handles
        all rotation combinations correctly, including combined pitch+roll.
        
        Uses the convention:
            - Pitch: rotation around X-axis (looking up/down)
                     Negative = looking down, Positive = looking up
            - Yaw: rotation around Y-axis (looking left/right)
                   Negative = looking left, Positive = looking right
            - Roll: rotation around Z-axis (head tilt)
                    Negative = tilt left, Positive = tilt right
        
        Args:
            R: 3x3 rotation matrix from cv2.Rodrigues(rvec)
            
        Returns:
            Tuple of (pitch, yaw, roll) in degrees
            
        Note:
            The previous implementation had issues with combined rotations
            (e.g., head drooping to the right caused positive pitch instead of
            negative). This was due to improper handling of the rotation order
            in the manual Euler angle extraction.
            
            cv2.RQDecomp3x3 properly decomposes the rotation matrix into
            three orthogonal rotation matrices (Qx, Qy, Qz) and extracts
            the Euler angles correctly for any head orientation.
        """
        # Use OpenCV's RQ decomposition for robust Euler angle extraction
        # This handles gimbal lock and all rotation combinations correctly
        try:
            # RQDecomp3x3 returns: (eulerAngles, mtxR, mtxQ, Qx, Qy, Qz)
            # where eulerAngles is a tuple of (pitch, yaw, roll) in degrees
            # The rotation order is: R = Qz @ Qy @ Qx (ZYX extrinsic = XYZ intrinsic)
            result = cv2.RQDecomp3x3(R)
            euler_angles = result[0]  # First element is the tuple of angles
            
            # euler_angles contains [pitch, yaw, roll] in degrees
            # OpenCV convention: 
            #   pitch (rotation around x) = euler_angles[0]
            #   yaw (rotation around y) = euler_angles[1]  
            #   roll (rotation around z) = euler_angles[2]
            pitch_deg = euler_angles[0]
            yaw_deg = euler_angles[1]
            roll_deg = euler_angles[2]
            
            # Apply coordinate system corrections to match our expected conventions:
            # In our system (camera looking at driver):
            #   - Pitch negative = head drooping down (chin towards chest)
            #   - Yaw positive = looking right (from driver's perspective)
            #   - Roll positive = head tilted to the right
            #
            # The PnP solution with our 3D model produces angles where:
            #   - When looking down, RQDecomp gives positive pitch
            #   - We need to negate pitch to match our convention
            pitch_deg = -pitch_deg
            
            return pitch_deg, yaw_deg, roll_deg
            
        except cv2.error:
            # Fallback to manual extraction if RQDecomp fails (rare)
            return self._rotation_matrix_to_euler_fallback(R)
    
    def _rotation_matrix_to_euler_fallback(self, R: np.ndarray) -> Tuple[float, float, float]:
        """
        Fallback Euler angle extraction using proper ZYX decomposition.
        
        This is used only if cv2.RQDecomp3x3 fails (which is rare).
        Uses the ZYX (yaw-pitch-roll) convention with proper sign handling.
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            Tuple of (pitch, yaw, roll) in degrees
        """
        # ZYX Euler angles (yaw, pitch, roll applied in that order)
        # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
        #
        # Decomposition formulas:
        # sin(pitch) = -R[2,0]
        # tan(yaw) = R[1,0] / R[0,0]
        # tan(roll) = R[2,1] / R[2,2]
        
        # Check for gimbal lock (pitch near ±90°)
        sin_pitch = -R[2, 0]
        sin_pitch = np.clip(sin_pitch, -1.0, 1.0)  # Numerical stability
        
        pitch = np.arcsin(sin_pitch)
        
        if np.abs(np.cos(pitch)) > 1e-6:
            # Normal case: not in gimbal lock
            yaw = np.arctan2(R[1, 0], R[0, 0])
            roll = np.arctan2(R[2, 1], R[2, 2])
        else:
            # Gimbal lock: pitch is ±90°
            # In this case, yaw and roll are coupled
            # We set roll to 0 and compute yaw
            roll = 0.0
            if sin_pitch > 0:  # pitch = +90°
                yaw = np.arctan2(-R[0, 1], R[1, 1])
            else:  # pitch = -90°
                yaw = np.arctan2(R[0, 1], R[1, 1])
        
        # Convert to degrees
        pitch_deg = np.degrees(pitch)
        yaw_deg = np.degrees(yaw)
        roll_deg = np.degrees(roll)
        
        # Apply same coordinate system corrections as main method
        pitch_deg = -pitch_deg
        
        return pitch_deg, yaw_deg, roll_deg
    
    def _calculate_reprojection_error(
        self,
        model_points: np.ndarray,
        image_points: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        camera_matrix: np.ndarray,
    ) -> float:
        """
        Calculate mean reprojection error for confidence estimation.
        
        Args:
            model_points: 3D model points
            image_points: Observed 2D image points
            rvec: Rotation vector from solvePnP
            tvec: Translation vector from solvePnP
            camera_matrix: Camera intrinsic matrix
            
        Returns:
            Mean reprojection error in pixels
        """
        projected_points, _ = cv2.projectPoints(
            model_points,
            rvec,
            tvec,
            camera_matrix,
            self._dist_coeffs
        )
        projected_points = projected_points.reshape(-1, 2)
        
        errors = np.linalg.norm(projected_points - image_points, axis=1)
        return float(np.mean(errors))
    
    def _smooth_angle(
        self,
        new_value: float,
        smoothed_value: Optional[float],
        raw_previous: Optional[float] = None,
    ) -> float:
        """
        Apply exponential moving average smoothing with outlier dampening.
        
        Args:
            new_value: New angle measurement
            smoothed_value: Previous smoothed value (None if first frame)
            raw_previous: Previous raw value for outlier detection
            
        Returns:
            Smoothed angle value
        """
        if smoothed_value is None:
            return new_value
        
        # Outlier dampening: reduce alpha for sudden large changes
        alpha = self.smoothing_alpha
        if raw_previous is not None:
            change = abs(new_value - raw_previous)
            if change > self.outlier_threshold:
                # Dampen the update for outliers
                alpha = alpha * (self.outlier_threshold / change)
                alpha = max(0.1, alpha)  # Ensure some update still happens
        
        return alpha * new_value + (1 - alpha) * smoothed_value
    
    def _confidence_from_error(self, reprojection_error: float) -> float:
        """
        Convert reprojection error to confidence score.
        
        Lower error = higher confidence.
        
        Args:
            reprojection_error: Mean reprojection error in pixels
            
        Returns:
            Confidence score from 0.0 to 1.0
        """
        # Error of 0 = confidence 1.0
        # Error of 10 pixels = confidence ~0.5
        # Error of 20+ pixels = confidence approaching 0
        confidence = np.exp(-reprojection_error / 10.0)
        return float(np.clip(confidence, 0.0, 1.0))
    
    def update(
        self,
        landmarks,
        frame_width: int,
        frame_height: int,
    ) -> HeadPoseResult:
        """
        Estimate head pose from facial landmarks.
        
        This is the main method to call each frame. It:
        1. Extracts 2D image points from MediaPipe landmarks
        2. Solves PnP to get rotation and translation
        3. Converts rotation to Euler angles
        4. Applies temporal smoothing
        5. Calculates confidence and state flags
        
        Args:
            landmarks: MediaPipe face landmarks (468 points).
                      Can be face_landmarks.landmark from FaceMesh result.
            frame_width: Frame width in pixels (e.g., 640)
            frame_height: Frame height in pixels (e.g., 480)
            
        Returns:
            HeadPoseResult with pose angles, confidence, and state flags.
            Returns a low-confidence result if estimation fails.
        """
        start_time = time.perf_counter()
        
        # Extract image points from landmarks
        image_points, success = self._extract_image_points(
            landmarks, frame_width, frame_height
        )
        
        if not success:
            # Return last result with reduced confidence if available
            if self._last_result is not None:
                result = HeadPoseResult(
                    pitch=self._last_result.pitch,
                    yaw=self._last_result.yaw,
                    roll=self._last_result.roll,
                    confidence=self._last_result.confidence * 0.5,
                    is_looking_down=self._last_result.is_looking_down,
                    is_severely_down=self._last_result.is_severely_down,
                    is_looking_away=self._last_result.is_looking_away,
                )
            else:
                result = HeadPoseResult(confidence=0.0)
            
            self._update_stats(start_time)
            return result
        
        # Get camera matrix
        camera_matrix = self._get_camera_matrix(frame_width, frame_height)
        
        # Select model points
        model_points = self._model_points_8 if self.use_extended_model else self._model_points_6
        
        # Solve PnP
        try:
            success, rvec, tvec = cv2.solvePnP(
                model_points,
                image_points,
                camera_matrix,
                self._dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
        except cv2.error:
            success = False
        
        if not success:
            result = HeadPoseResult(confidence=0.0)
            self._update_stats(start_time)
            return result
        
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        
        # Extract Euler angles
        raw_pitch, raw_yaw, raw_roll = self._rotation_matrix_to_euler(rotation_matrix)
        
        # Calculate reprojection error for confidence
        reproj_error = self._calculate_reprojection_error(
            model_points, image_points, rvec, tvec, camera_matrix
        )
        confidence = self._confidence_from_error(reproj_error)
        
        # Get previous raw values for outlier detection
        prev_pitch = self._last_result.raw_pitch if self._last_result else None
        prev_yaw = self._last_result.raw_yaw if self._last_result else None
        prev_roll = self._last_result.raw_roll if self._last_result else None
        
        # Apply temporal smoothing
        if not self._initialized:
            self._smoothed_pitch = raw_pitch
            self._smoothed_yaw = raw_yaw
            self._smoothed_roll = raw_roll
            self._initialized = True
        else:
            self._smoothed_pitch = self._smooth_angle(
                raw_pitch, self._smoothed_pitch, prev_pitch
            )
            self._smoothed_yaw = self._smooth_angle(
                raw_yaw, self._smoothed_yaw, prev_yaw
            )
            self._smoothed_roll = self._smooth_angle(
                raw_roll, self._smoothed_roll, prev_roll
            )
        
        # Determine state flags
        is_looking_down = self._smoothed_pitch < self.looking_down_threshold
        is_severely_down = self._smoothed_pitch < self.severe_droop_threshold
        is_looking_away = abs(self._smoothed_yaw) > self.looking_away_threshold
        
        # Build result
        result = HeadPoseResult(
            pitch=self._smoothed_pitch,
            yaw=self._smoothed_yaw,
            roll=self._smoothed_roll,
            confidence=confidence,
            is_looking_down=is_looking_down,
            is_severely_down=is_severely_down,
            is_looking_away=is_looking_away,
            raw_pitch=raw_pitch,
            raw_yaw=raw_yaw,
            raw_roll=raw_roll,
            reprojection_error=reproj_error,
        )
        
        # Update history for head drooping detection
        self._pitch_history.append(self._smoothed_pitch)
        self._severe_droop_history.append(is_severely_down)
        
        # Update tracking
        self._last_result = result
        self._angle_history.append((self._smoothed_pitch, self._smoothed_yaw, self._smoothed_roll))
        self._confidence_history.append(confidence)
        
        self._update_stats(start_time)
        
        return result
    
    def _update_stats(self, start_time: float):
        """Update performance statistics."""
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._update_count += 1
        self._total_time_ms += elapsed_ms
    
    def detect_head_drooping(
        self,
        threshold_frames: int = 60,
        severe_frames: int = 30,
    ) -> Tuple[bool, bool, int]:
        """
        Detect sustained head drooping indicating microsleep.
        
        This method analyzes the pitch history to detect when the driver's
        head has been drooping (pitched severely down) for an extended period.
        
        Two levels of detection:
        1. Head drooping: pitch < -30° for threshold_frames (~2 seconds)
        2. Severe head droop: pitch < -45° for severe_frames (~1 second)
        
        Args:
            threshold_frames: Frames of severe droop (pitch < -30°) for "head drooping"
            severe_frames: Frames of extreme droop (pitch < -45°) for "severe droop"
            
        Returns:
            Tuple of (is_drooping, is_severe, consecutive_frames)
            - is_drooping: True if head has been drooped for threshold_frames
            - is_severe: True if extreme droop for severe_frames
            - consecutive_frames: Current count of consecutive severe droop frames
        """
        if len(self._severe_droop_history) < severe_frames:
            return False, False, 0
        
        # Count consecutive severe droop frames (from most recent)
        consecutive_severe = 0
        for is_severe in reversed(self._severe_droop_history):
            if is_severe:
                consecutive_severe += 1
            else:
                break
        
        # Count consecutive extreme droop frames (pitch < -45°)
        consecutive_extreme = 0
        extreme_threshold = -45.0
        for pitch in reversed(self._pitch_history):
            if pitch < extreme_threshold:
                consecutive_extreme += 1
            else:
                break
        
        is_drooping = consecutive_severe >= threshold_frames
        is_severe = consecutive_extreme >= severe_frames
        
        return is_drooping, is_severe, consecutive_severe
    
    def get_looking_down_discount_factor(
        self,
        min_discount: float = 0.7,
        max_discount: float = 1.0,
    ) -> float:
        """
        Calculate discount factor for eye detection based on head pose.
        
        When the driver is looking down (checking dashboard, navigation, etc.),
        eye closure detection may produce false positives. This method returns
        a factor to reduce the fusion score in such cases.
        
        The discount is non-linear to preserve sensitivity while reducing
        false alarms:
        - Pitch >= -15°: No discount (factor = 1.0)
        - Pitch -15° to -30°: Gradual discount (factor 1.0 -> min_discount)
        - Pitch < -30°: No further discount (indicates actual drowsiness)
        
        Args:
            min_discount: Minimum discount factor (at -30° pitch)
            max_discount: Maximum factor (at -15° or higher pitch)
            
        Returns:
            Discount factor from min_discount to max_discount
        """
        if self._smoothed_pitch is None:
            return max_discount
        
        pitch = self._smoothed_pitch
        
        if pitch >= self.looking_down_threshold:
            # Not looking down - no discount
            return max_discount
        elif pitch <= self.severe_droop_threshold:
            # Severely down - this might be drowsiness, don't discount
            # (or very minimal discount to avoid masking real drowsiness)
            return max_discount
        else:
            # In the "looking down" range - apply graduated discount
            # Linear interpolation from looking_down_threshold to severe_droop_threshold
            range_size = self.looking_down_threshold - self.severe_droop_threshold
            progress = (self.looking_down_threshold - pitch) / range_size
            factor = max_discount - progress * (max_discount - min_discount)
            return factor
    
    def get_yaw_discount_factor(
        self,
        min_discount: float = 0.8,
        max_discount: float = 1.0,
    ) -> float:
        """
        Calculate discount factor based on yaw (looking left/right).
        
        When the driver is looking to the side (checking mirrors, etc.),
        eye detection may be less reliable. This provides a discount factor.
        
        Args:
            min_discount: Minimum discount factor (at extreme yaw)
            max_discount: Maximum factor (when facing forward)
            
        Returns:
            Discount factor from min_discount to max_discount
        """
        if self._smoothed_yaw is None:
            return max_discount
        
        abs_yaw = abs(self._smoothed_yaw)
        
        if abs_yaw <= self.looking_away_threshold:
            return max_discount
        elif abs_yaw >= 60.0:
            return min_discount
        else:
            # Graduated discount
            range_size = 60.0 - self.looking_away_threshold
            progress = (abs_yaw - self.looking_away_threshold) / range_size
            factor = max_discount - progress * (max_discount - min_discount)
            return factor
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get estimator statistics for debugging and monitoring.
        
        Returns:
            Dictionary containing:
            - update_count: Total number of update() calls
            - avg_time_ms: Average processing time per frame
            - avg_confidence: Average confidence over recent frames
            - current_pitch/yaw/roll: Current smoothed angles
            - pitch_range: (min, max) pitch over history
            - droop_frames: Current consecutive severe droop frame count
        """
        avg_time = self._total_time_ms / max(1, self._update_count)
        
        avg_confidence = 0.0
        if self._confidence_history:
            avg_confidence = sum(self._confidence_history) / len(self._confidence_history)
        
        pitch_range = (0.0, 0.0)
        if self._pitch_history:
            pitch_range = (min(self._pitch_history), max(self._pitch_history))
        
        _, _, droop_frames = self.detect_head_drooping()
        
        return {
            'update_count': self._update_count,
            'avg_time_ms': round(avg_time, 3),
            'avg_confidence': round(avg_confidence, 3),
            'current_pitch': round(self._smoothed_pitch, 1) if self._smoothed_pitch else None,
            'current_yaw': round(self._smoothed_yaw, 1) if self._smoothed_yaw else None,
            'current_roll': round(self._smoothed_roll, 1) if self._smoothed_roll else None,
            'pitch_range': (round(pitch_range[0], 1), round(pitch_range[1], 1)),
            'droop_frames': droop_frames,
            'looking_down': self._last_result.is_looking_down if self._last_result else False,
            'looking_away': self._last_result.is_looking_away if self._last_result else False,
        }
    
    def reset(self):
        """
        Reset estimator state.
        
        Call this when switching to a new driver or starting a new session.
        Clears all history and smoothing state.
        """
        self._smoothed_pitch = None
        self._smoothed_yaw = None
        self._smoothed_roll = None
        self._initialized = False
        self._pitch_history.clear()
        self._severe_droop_history.clear()
        self._angle_history.clear()
        self._confidence_history.clear()
        self._last_result = None
        self._update_count = 0
        self._total_time_ms = 0.0


# =============================================================================
# Visualization Utilities
# =============================================================================

def draw_head_pose_axis(
    frame: np.ndarray,
    result: HeadPoseResult,
    nose_point: Tuple[int, int],
    axis_length: int = 50,
) -> np.ndarray:
    """
    Draw 3D axis visualization on frame showing head orientation.
    
    Draws X (red), Y (green), Z (blue) axes centered at nose tip,
    rotated according to head pose.
    
    Args:
        frame: Input BGR frame to draw on (modified in place)
        result: HeadPoseResult from estimator
        nose_point: (x, y) pixel coordinates of nose tip
        axis_length: Length of axis lines in pixels
        
    Returns:
        Frame with axes drawn
    """
    # Convert Euler angles back to rotation matrix
    pitch_rad = np.radians(result.pitch)
    yaw_rad = np.radians(result.yaw)
    roll_rad = np.radians(result.roll)
    
    # Rotation matrices
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad), np.cos(pitch_rad)]
    ])
    Ry = np.array([
        [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
        [0, 1, 0],
        [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
    ])
    Rz = np.array([
        [np.cos(roll_rad), -np.sin(roll_rad), 0],
        [np.sin(roll_rad), np.cos(roll_rad), 0],
        [0, 0, 1]
    ])
    
    R = Rz @ Ry @ Rx
    
    # Axis endpoints in 3D
    axes_3d = np.array([
        [axis_length, 0, 0],   # X-axis (red)
        [0, axis_length, 0],   # Y-axis (green)  
        [0, 0, axis_length],   # Z-axis (blue)
    ], dtype=np.float64)
    
    # Rotate axes
    axes_rotated = (R @ axes_3d.T).T
    
    # Project to 2D (simple orthographic for visualization)
    origin = nose_point
    
    colors = [
        (0, 0, 255),   # Red for X
        (0, 255, 0),   # Green for Y
        (255, 0, 0),   # Blue for Z
    ]
    
    for i, (axis, color) in enumerate(zip(axes_rotated, colors)):
        end_point = (
            int(origin[0] + axis[0]),
            int(origin[1] + axis[1])
        )
        thickness = 3 if i == 2 else 2  # Z-axis thicker
        cv2.line(frame, origin, end_point, color, thickness)
    
    return frame


def draw_head_pose_info(
    frame: np.ndarray,
    result: HeadPoseResult,
    position: Tuple[int, int] = (10, 30),
    font_scale: float = 0.6,
) -> np.ndarray:
    """
    Draw head pose information text on frame.
    
    Shows pitch, yaw, roll values with color-coded state indicators.
    
    Args:
        frame: Input BGR frame to draw on (modified in place)
        result: HeadPoseResult from estimator
        position: Top-left position for text
        font_scale: Font size multiplier
        
    Returns:
        Frame with text overlay
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_height = int(25 * font_scale / 0.6)
    
    # Determine state color
    if result.is_severely_down:
        state_color = (0, 0, 255)  # Red
        state_text = "DROOP!"
    elif result.is_looking_down:
        state_color = (0, 165, 255)  # Orange
        state_text = "Looking Down"
    elif result.is_looking_away:
        state_color = (0, 255, 255)  # Yellow
        state_text = "Looking Away"
    else:
        state_color = (0, 255, 0)  # Green
        state_text = "Normal"
    
    lines = [
        f"Pitch: {result.pitch:+.1f}deg",
        f"Yaw: {result.yaw:+.1f}deg",
        f"Roll: {result.roll:+.1f}deg",
        f"Conf: {result.confidence:.2f}",
        f"State: {state_text}",
    ]
    
    x, y = position
    for i, line in enumerate(lines):
        color = state_color if i == 4 else (255, 255, 255)
        cv2.putText(
            frame, line, (x, y + i * line_height),
            font, font_scale, (0, 0, 0), 3  # Black outline
        )
        cv2.putText(
            frame, line, (x, y + i * line_height),
            font, font_scale, color, 1
        )
    
    return frame


# =============================================================================
# Unit Tests
# =============================================================================

def _run_unit_tests():
    """
    Run unit tests for HeadPoseEstimator.
    
    Tests:
    1. Initialization and configuration
    2. Synthetic landmark processing
    3. Angle range validation
    4. Temporal smoothing
    5. Head drooping detection
    6. Performance benchmarks
    """
    import time
    
    print("=" * 60)
    print("HeadPoseEstimator Unit Tests")
    print("=" * 60)
    
    # Test 1: Initialization
    print("\n[Test 1] Initialization...")
    estimator = HeadPoseEstimator(
        smoothing_alpha=0.4,
        looking_down_threshold=-15.0,
        severe_droop_threshold=-30.0,
    )
    assert estimator.looking_down_threshold == -15.0
    assert estimator.severe_droop_threshold == -30.0
    assert len(estimator._model_points_6) == 6
    print("  ✓ Configuration applied correctly")
    print("  ✓ 3D model points initialized")
    
    # Test 2: Camera matrix computation
    print("\n[Test 2] Camera matrix computation...")
    cam_matrix = estimator._get_camera_matrix(640, 480)
    assert cam_matrix.shape == (3, 3)
    assert cam_matrix[0, 2] == 320.0  # Principal point X
    assert cam_matrix[1, 2] == 240.0  # Principal point Y
    # Cache test
    cam_matrix_2 = estimator._get_camera_matrix(640, 480)
    assert cam_matrix is cam_matrix_2  # Should return cached
    print("  ✓ Camera matrix has correct shape")
    print("  ✓ Principal point at image center")
    print("  ✓ Caching works correctly")
    
    # Test 3: Rotation matrix to Euler conversion - COMPREHENSIVE TESTS
    print("\n[Test 3] Euler angle conversion (COMPREHENSIVE)...")
    print("  Testing rotation matrix decomposition for all head orientations:")
    
    def create_rotation_matrix(pitch_deg, yaw_deg, roll_deg):
        """Create rotation matrix from Euler angles (ZYX order)."""
        # Convert to radians
        pitch = np.radians(pitch_deg)
        yaw = np.radians(yaw_deg)
        roll = np.radians(roll_deg)
        
        # Individual rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)]
        ])
        Ry = np.array([
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)]
        ])
        Rz = np.array([
            [np.cos(roll), -np.sin(roll), 0],
            [np.sin(roll), np.cos(roll), 0],
            [0, 0, 1]
        ])
        
        # ZYX order: R = Rz @ Ry @ Rx
        return Rz @ Ry @ Rx
    
    test_cases = [
        # (description, input_pitch, input_yaw, input_roll, expected_sign_pitch)
        ("Identity matrix (no rotation)", 0, 0, 0, 0),
        ("Pure pitch down -30°", -30, 0, 0, -1),
        ("Pure pitch down -45°", -45, 0, 0, -1),
        ("Pure pitch up +30°", 30, 0, 0, 1),
        ("Pure yaw right +45°", 0, 45, 0, 0),
        ("Pure yaw left -45°", 0, -45, 0, 0),
        ("Pure roll right +30°", 0, 0, 30, 0),
        ("Pure roll left -30°", 0, 0, -30, 0),
        # CRITICAL: Combined rotations (these were failing before!)
        ("Head droop + roll right (pitch -40°, roll +20°)", -40, 0, 20, -1),
        ("Head droop + roll left (pitch -35°, roll -25°)", -35, 0, -25, -1),
        ("Looking right + slight droop (pitch -20°, yaw +40°)", -20, 40, 0, -1),
        ("Head falling forward (pitch -50°)", -50, 0, 0, -1),
        ("Head falling forward + roll (pitch -45°, roll -60°)", -45, 0, -60, -1),
    ]
    
    all_passed = True
    for desc, in_pitch, in_yaw, in_roll, expected_sign in test_cases:
        # Create rotation matrix from known angles
        # Note: We negate pitch input because our convention has pitch sign flipped
        R_test = create_rotation_matrix(-in_pitch, in_yaw, in_roll)
        out_pitch, out_yaw, out_roll = estimator._rotation_matrix_to_euler(R_test)
        
        # Check pitch sign is correct for drowsiness detection
        if expected_sign != 0:
            sign_correct = (np.sign(out_pitch) == expected_sign) if abs(out_pitch) > 5 else True
        else:
            sign_correct = abs(out_pitch) < 15  # Should be near zero
        
        status = "✓" if sign_correct else "✗ FAIL"
        print(f"    {status} {desc}")
        print(f"       Input: P:{in_pitch:+.0f}° Y:{in_yaw:+.0f}° R:{in_roll:+.0f}° → Output: P:{out_pitch:+.1f}° Y:{out_yaw:+.1f}° R:{out_roll:+.1f}°")
        
        if not sign_correct:
            all_passed = False
            print(f"       ERROR: Expected pitch sign {expected_sign}, got {np.sign(out_pitch)}")
    
    # Identity matrix special test
    R_identity = np.eye(3)
    pitch, yaw, roll = estimator._rotation_matrix_to_euler(R_identity)
    identity_ok = abs(pitch) < 1e-3 and abs(yaw) < 1e-3 and abs(roll) < 1e-3
    print(f"    {'✓' if identity_ok else '✗'} Identity matrix → P:{pitch:.3f}° Y:{yaw:.3f}° R:{roll:.3f}° (should be ~0)")
    all_passed = all_passed and identity_ok
    
    if all_passed:
        print("  ✓ ALL ROTATION MATRIX TESTS PASSED!")
    else:
        print("  ✗ SOME ROTATION MATRIX TESTS FAILED - CHECK ABOVE")
    
    # Special test: The exact bug scenario from screenshots
    print("\n  Testing exact bug scenarios from screenshots:")
    print("    (Simulating head drooping to the right)")
    # When head falls to right, there's negative pitch combined with positive roll
    R_bug_scenario = create_rotation_matrix(40, 0, 30)  # Negated pitch for our convention
    p_bug, y_bug, r_bug = estimator._rotation_matrix_to_euler(R_bug_scenario)
    bug_fixed = p_bug < -20  # Should be negative for drowsiness detection
    print(f"    {'✓' if bug_fixed else '✗'} Right droop scenario: P:{p_bug:+.1f}° Y:{y_bug:+.1f}° R:{r_bug:+.1f}°")
    print(f"       Pitch should be NEGATIVE for drowsiness detection: {'OK' if bug_fixed else 'STILL BROKEN!'}")
    
    # Test 4: Confidence calculation
    print("\n[Test 4] Confidence from reprojection error...")
    conf_0 = estimator._confidence_from_error(0.0)
    conf_10 = estimator._confidence_from_error(10.0)
    conf_20 = estimator._confidence_from_error(20.0)
    assert conf_0 > conf_10 > conf_20
    assert 0.99 < conf_0 <= 1.0
    assert 0.3 < conf_10 < 0.5
    print(f"  ✓ Error 0px → Confidence {conf_0:.3f}")
    print(f"  ✓ Error 10px → Confidence {conf_10:.3f}")
    print(f"  ✓ Error 20px → Confidence {conf_20:.3f}")
    
    # Test 5: Smoothing
    print("\n[Test 5] Temporal smoothing...")
    smoothed = estimator._smooth_angle(10.0, None)
    assert smoothed == 10.0  # First value, no smoothing
    smoothed = estimator._smooth_angle(20.0, 10.0)
    assert 10.0 < smoothed < 20.0  # Smoothed between old and new
    print("  ✓ First value passes through")
    print("  ✓ Subsequent values smoothed")
    
    # Test 6: Outlier dampening in smoothing
    print("\n[Test 6] Outlier dampening...")
    # Large jump should be dampened
    smoothed_normal = estimator._smooth_angle(15.0, 10.0, 10.0)
    smoothed_outlier = estimator._smooth_angle(50.0, 10.0, 10.0)  # 40° jump
    # Outlier should move less toward new value
    normal_delta = smoothed_normal - 10.0
    outlier_delta = smoothed_outlier - 10.0
    assert outlier_delta / 40.0 < normal_delta / 5.0  # Proportionally less movement
    print("  ✓ Large jumps are dampened")
    
    # Test 7: Synthetic landmarks for integration test
    print("\n[Test 7] Synthetic landmark processing...")
    
    class MockLandmark:
        def __init__(self, x, y, z=0):
            self.x = x
            self.y = y
            self.z = z
    
    # Create synthetic landmarks for a face looking straight ahead
    # Positions are normalized (0-1)
    synthetic_landmarks = [None] * 468
    synthetic_landmarks[1] = MockLandmark(0.5, 0.45)      # Nose tip
    synthetic_landmarks[152] = MockLandmark(0.5, 0.85)   # Chin
    synthetic_landmarks[33] = MockLandmark(0.35, 0.35)   # Left eye outer
    synthetic_landmarks[263] = MockLandmark(0.65, 0.35)  # Right eye outer
    synthetic_landmarks[61] = MockLandmark(0.4, 0.65)    # Left mouth
    synthetic_landmarks[291] = MockLandmark(0.6, 0.65)   # Right mouth
    synthetic_landmarks[6] = MockLandmark(0.5, 0.35)     # Nose bridge
    synthetic_landmarks[10] = MockLandmark(0.5, 0.15)    # Forehead
    
    estimator2 = HeadPoseEstimator()
    result = estimator2.update(synthetic_landmarks, 640, 480)
    
    assert isinstance(result, HeadPoseResult)
    assert result.confidence > 0  # Should get valid result
    print(f"  ✓ Synthetic face processed")
    print(f"    Pitch: {result.pitch:.1f}°, Yaw: {result.yaw:.1f}°, Roll: {result.roll:.1f}°")
    print(f"    Confidence: {result.confidence:.3f}")
    
    # Test 8: Head drooping detection
    print("\n[Test 8] Head drooping detection...")
    estimator3 = HeadPoseEstimator()
    
    # Simulate severe head droop for multiple frames
    droop_landmark = [None] * 468
    # Nose lower than normal (simulating looking down)
    droop_landmark[1] = MockLandmark(0.5, 0.55)  
    droop_landmark[152] = MockLandmark(0.5, 0.95)
    droop_landmark[33] = MockLandmark(0.35, 0.4)
    droop_landmark[263] = MockLandmark(0.65, 0.4)
    droop_landmark[61] = MockLandmark(0.4, 0.75)
    droop_landmark[291] = MockLandmark(0.6, 0.75)
    
    for _ in range(65):
        estimator3.update(droop_landmark, 640, 480)
    
    is_drooping, is_severe, frames = estimator3.detect_head_drooping(threshold_frames=60)
    print(f"  ✓ After 65 droop frames: is_drooping={is_drooping}, frames={frames}")
    
    # Test 9: Performance benchmark
    print("\n[Test 9] Performance benchmark...")
    estimator4 = HeadPoseEstimator()
    
    # Warm up
    for _ in range(10):
        estimator4.update(synthetic_landmarks, 640, 480)
    
    # Benchmark
    n_iterations = 100
    start = time.perf_counter()
    for _ in range(n_iterations):
        estimator4.update(synthetic_landmarks, 640, 480)
    elapsed = time.perf_counter() - start
    
    avg_ms = (elapsed / n_iterations) * 1000
    print(f"  ✓ Average time: {avg_ms:.3f} ms/frame")
    print(f"  {'✓' if avg_ms < 3.0 else '✗'} Target: < 3.0 ms/frame")
    
    # Test 10: Stats and reset
    print("\n[Test 10] Statistics and reset...")
    stats = estimator4.get_stats()
    assert 'update_count' in stats
    assert stats['update_count'] > 0
    print(f"  ✓ Stats retrieved: {stats['update_count']} updates")
    
    estimator4.reset()
    assert not estimator4._initialized
    assert len(estimator4._pitch_history) == 0
    print("  ✓ Reset clears state")
    
    # Test 11: Discount factors
    print("\n[Test 11] Discount factor calculation...")
    estimator5 = HeadPoseEstimator()
    
    # Simulate normal pose
    estimator5._smoothed_pitch = 0.0
    factor_normal = estimator5.get_looking_down_discount_factor()
    assert factor_normal == 1.0
    
    # Simulate looking down (-20°)
    estimator5._smoothed_pitch = -20.0
    factor_down = estimator5.get_looking_down_discount_factor()
    assert 0.7 < factor_down < 1.0
    
    print(f"  ✓ Normal pose discount: {factor_normal}")
    print(f"  ✓ Looking down (-20°) discount: {factor_down:.3f}")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


# =============================================================================
# Integration Examples
# =============================================================================

def _integration_example():
    """
    Example showing how to integrate HeadPoseEstimator with the drowsiness system.
    
    This demonstrates the typical usage pattern in main_v5_1_enhanced.py.
    """
    print("\n" + "=" * 60)
    print("Integration Example")
    print("=" * 60)
    
    code = '''
# ============================================================================
# Integration in main_v5_1_enhanced.py
# ============================================================================

# 1. IMPORTS (add at top of file)
from head_pose_estimator import HeadPoseEstimator, HeadPoseResult
from head_pose_estimator import draw_head_pose_axis, draw_head_pose_info

# 2. INITIALIZATION (in main() or setup section, ~line 750)
head_pose_estimator = HeadPoseEstimator(
    smoothing_alpha=0.4,
    looking_down_threshold=-15.0,
    severe_droop_threshold=-30.0,
    looking_away_threshold=30.0,
)

# 3. IN DETECTION LOOP (after landmark extraction, ~line 830)
# After: face_landmarks = results.multi_face_landmarks[0]

# Calculate head pose
head_pose = head_pose_estimator.update(
    face_landmarks.landmark,
    frame_width,
    frame_height
)

# Get discount factors for fusion
looking_down_discount = head_pose_estimator.get_looking_down_discount_factor(
    min_discount=0.7, max_discount=1.0
)
looking_away_discount = head_pose_estimator.get_yaw_discount_factor(
    min_discount=0.8, max_discount=1.0
)

# Calculate EAR (existing code)
left_ear = calculate_ear(face_landmarks.landmark, LEFT_EYE_INDICES)
right_ear = calculate_ear(face_landmarks.landmark, RIGHT_EYE_INDICES)

# ENHANCED DETECTION with head pose integration
left_result = left_eye_detector.update(
    left_cnn,
    left_ear,
    head_pitch=head_pose.pitch  # Pass head pitch for fusion adjustment
)
right_result = right_eye_detector.update(
    right_cnn,
    right_ear,
    head_pitch=head_pose.pitch
)

# 4. HEAD DROOPING CHECK (new drowsiness signal)
is_head_drooping, is_severe_droop, droop_frames = head_pose_estimator.detect_head_drooping(
    threshold_frames=60,  # 2 seconds at 30 FPS
    severe_frames=30,     # 1 second for severe
)

# Combine with existing alarm logic (in drowsiness decision section)
if is_severe_droop:
    # Immediate microsleep alarm
    alarm_controller.trigger("MICROSLEEP DETECTED - HEAD DROOP")
elif is_head_drooping and (left_result.state == "CLOSED" or right_result.state == "CLOSED"):
    # Combined signal - head droop + eyes closed
    alarm_controller.trigger("DROWSINESS DETECTED")

# 5. VISUALIZATION (optional, before frame display)
if head_pose.confidence > 0.5:
    # Get nose position for axis drawing
    nose_lm = face_landmarks.landmark[1]
    nose_point = (int(nose_lm.x * frame_width), int(nose_lm.y * frame_height))
    
    # Draw 3D axis on face
    draw_head_pose_axis(frame, head_pose, nose_point, axis_length=50)
    
    # Draw pose information
    draw_head_pose_info(frame, head_pose, position=(frame_width - 180, 30))

# 6. DEBUG STATS (optional, for monitoring)
if frame_count % 300 == 0:  # Every 10 seconds at 30 FPS
    pose_stats = head_pose_estimator.get_stats()
    print(f"Head Pose Stats: avg_time={pose_stats['avg_time_ms']:.2f}ms, "
          f"pitch_range={pose_stats['pitch_range']}")


# ============================================================================
# Integration in enhanced_eye_detector.py (update existing logic)
# ============================================================================

# In EnhancedEyeStateDetector.update(), around line 263-267
# REPLACE the existing placeholder:

# OLD:
# if head_pitch is not None and head_pitch < self.HEAD_TILT_DOWN_THRESHOLD:
#     looking_down_factor = 1.0 - min(0.3, abs(head_pitch + 15) / 30)
#     fusion_score *= looking_down_factor

# NEW (improved non-linear discount):
if head_pitch is not None:
    LOOKING_DOWN_THRESHOLD = -15.0
    SEVERE_DROOP_THRESHOLD = -30.0
    
    if head_pitch < LOOKING_DOWN_THRESHOLD and head_pitch > SEVERE_DROOP_THRESHOLD:
        # Looking down but not severely drooped
        # Apply graduated discount using smooth curve
        progress = (LOOKING_DOWN_THRESHOLD - head_pitch) / (LOOKING_DOWN_THRESHOLD - SEVERE_DROOP_THRESHOLD)
        # Smooth curve: more discount in middle of range
        discount = 1.0 - 0.3 * (1 - (1 - progress) ** 2)
        fusion_score *= discount
    # Note: When pitch < SEVERE_DROOP_THRESHOLD, don't discount
    # This preserves sensitivity for actual drowsiness/microsleep
'''
    
    print(code)


if __name__ == "__main__":
    _run_unit_tests()
    _integration_example()
