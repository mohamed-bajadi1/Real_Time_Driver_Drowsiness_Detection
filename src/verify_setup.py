"""
Setup Verification Script
Tests all dependencies and camera availability
"""

import sys

def check_import(module_name, display_name=None):
    """Check if a module can be imported"""
    if display_name is None:
        display_name = module_name
    
    try:
        __import__(module_name)
        print(f"✅ {display_name} - OK")
        return True
    except ImportError as e:
        print(f"❌ {display_name} - FAILED")
        print(f"   Error: {e}")
        return False

def check_camera():
    """Check if camera is available"""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print(f"✅ Camera (index 0) - OK")
                print(f"   Resolution: {frame.shape[1]}x{frame.shape[0]}")
                return True
        print(f"❌ Camera (index 0) - FAILED (Cannot open)")
        return False
    except Exception as e:
        print(f"❌ Camera - ERROR: {e}")
        return False

def main():
    print("=" * 60)
    print("Driver Drowsiness Detection - Setup Verification")
    print("=" * 60)
    print()
    
    print("📦 Checking Python Dependencies...")
    print("-" * 60)
    
    dependencies = [
        ("cv2", "OpenCV"),
        ("mediapipe", "MediaPipe"),
        ("tensorflow", "TensorFlow"),
        ("numpy", "NumPy"),
        ("sklearn", "Scikit-learn"),
    ]
    
    results = []
    for module, name in dependencies:
        results.append(check_import(module, name))
    
    print()
    print("📷 Checking Camera...")
    print("-" * 60)
    camera_ok = check_camera()
    
    print()
    print("=" * 60)
    
    all_ok = all(results) and camera_ok
    
    if all_ok:
        print("✅ ALL CHECKS PASSED!")
        print("You can now run: python main.py")
    else:
        print("❌ SOME CHECKS FAILED")
        print("Please install missing dependencies:")
        print("   pip install -r requirements.txt")
    
    print("=" * 60)
    print()
    
    # Show versions
    print("📊 Installed Versions:")
    print("-" * 60)
    try:
        import cv2
        print(f"OpenCV: {cv2.__version__}")
    except: pass
    
    try:
        import mediapipe
        print(f"MediaPipe: {mediapipe.__version__}")
    except: pass
    
    try:
        import tensorflow as tf
        print(f"TensorFlow: {tf.__version__}")
    except: pass
    
    try:
        import numpy as np
        print(f"NumPy: {np.__version__}")
    except: pass
    
    print("=" * 60)
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
