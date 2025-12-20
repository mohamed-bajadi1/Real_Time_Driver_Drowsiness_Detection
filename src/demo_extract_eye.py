"""
Simple Demo: Test extract_eye() function on a single image
Perfect for quick verification before running full application
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# ============================================
# CONFIGURATION
# ============================================
MODEL_PATH = "C:\\Users\\hp\\Desktop\\Real_Time_Driver_Drowsiness_Detection\\models\\eye_state_classifier.h5"
EYE_IMG_SIZE = (64, 64)


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

    # Recreate the model architecture from training notebook
    # This MUST match train-eye-model.ipynb exactly!
    from keras.regularizers import l2

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
        # If that fails, try loading the whole model with custom handling
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

# Example test image paths (update these to match your data)
SAMPLE_CLOSED_EYE = "C:\\Users\\hp\\Desktop\\Real_Time_Driver_Drowsiness_Detection\\data\\eyes\\test\\Close\\closed_eye_0032.jpg_face_1_R.jpg"
SAMPLE_OPEN_EYE = "C:\\Users\\hp\\Desktop\\Real_Time_Driver_Drowsiness_Detection\\data\\eyes\\test\\Open\\Al_Leiter_0001_L.jpg"


# ============================================
# EXTRACT_EYE FUNCTION
# ============================================

def extract_eye(image_path):
    """
    Extract and preprocess eye image for CNN model
    
    This is the main function you need for Phase 2!
    
    Steps:
    1. Load image
    2. Convert to grayscale
    3. Resize to 64x64
    4. Normalize to [0, 1]
    5. Add dimensions for model input
    
    Args:
        image_path: Path to eye image
    
    Returns:
        preprocessed_image: Ready for model.predict()
        original_gray: For visualization
    """
    print(f"\n{'='*60}")
    print("STEP-BY-STEP: extract_eye() function")
    print(f"{'='*60}")
    
    # Step 1: Load image
    print("\n[1] Loading image...")
    img = cv2.imread(image_path)

    if img is None:
        print(f"   [ERROR] Could not read: {image_path}")
        return None, None

    print(f"   [OK] Image loaded: {img.shape}")
    print(f"      Shape: (height={img.shape[0]}, width={img.shape[1]}, channels={img.shape[2]})")
    
    # Step 2: Convert to grayscale
    print("\n[2] Converting to grayscale...")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"   [OK] Grayscale: {img_gray.shape}")
    print(f"      Shape: (height={img_gray.shape[0]}, width={img_gray.shape[1]})")

    # Step 3: Resize to 64x64
    print("\n[3] Resizing to 64x64...")
    img_resized = cv2.resize(img_gray, EYE_IMG_SIZE)
    print(f"   [OK] Resized: {img_resized.shape}")

    # Step 4: Normalize to [0, 1]
    print("\n[4] Normalizing to [0, 1]...")
    img_normalized = img_resized / 255.0
    print(f"   [OK] Normalized")
    print(f"      Min value: {img_normalized.min():.3f}")
    print(f"      Max value: {img_normalized.max():.3f}")

    # Step 5: Add channel dimension (64, 64) -> (64, 64, 1)
    print("\n[5] Adding channel dimension...")
    img_with_channel = np.expand_dims(img_normalized, axis=-1)
    print(f"   [OK] Channel added: {img_with_channel.shape}")
    print(f"      Shape: (height={img_with_channel.shape[0]}, width={img_with_channel.shape[1]}, channels={img_with_channel.shape[2]})")

    # Step 6: Add batch dimension (64, 64, 1) -> (1, 64, 64, 1)
    print("\n[6] Adding batch dimension...")
    img_batch = np.expand_dims(img_with_channel, axis=0)
    print(f"   [OK] Batch added: {img_batch.shape}")
    print(f"      Shape: (batch={img_batch.shape[0]}, height={img_batch.shape[1]}, width={img_batch.shape[2]}, channels={img_batch.shape[3]})")

    print(f"\n{'='*60}")
    print("[OK] Preprocessing complete!")
    print(f"   Final shape: {img_batch.shape}")
    print(f"   Ready for: model.predict(img_batch)")
    print(f"{'='*60}")
    
    return img_batch, img_gray


# ============================================
# PREDICTION FUNCTION
# ============================================

def predict_eye_state(model, preprocessed_image):
    """
    Predict eye state using the CNN model
    
    Args:
        model: Loaded Keras model
        preprocessed_image: Output from extract_eye() - shape (1, 64, 64, 1)
    
    Returns:
        prediction: Raw output (0-1)
        label: "OPEN" or "CLOSED"
        confidence: Percentage
    """
    if preprocessed_image is None:
        return None, None, None
    
    print(f"\n{'='*60}")
    print("PREDICTION")
    print(f"{'='*60}")

    # Get prediction
    print("\n[PREDICTING] Running model.predict()...")
    prediction = model.predict(preprocessed_image, verbose=0)[0][0]

    print(f"   Raw output: {prediction:.4f}")
    print(f"\n   Interpretation:")
    print(f"   - Output close to 0.0 -> Eye CLOSED")
    print(f"   - Output close to 1.0 -> Eye OPEN")
    print(f"   - Threshold: 0.5")

    # Determine class
    is_closed = prediction < 0.5
    label = "CLOSED" if is_closed else "OPEN"

    # Calculate confidence
    confidence = (1 - prediction) * 100 if is_closed else prediction * 100

    print(f"\n   [RESULT] Prediction: {label}")
    print(f"   [RESULT] Confidence: {confidence:.2f}%")
    
    return prediction, label, confidence


# ============================================
# DEMO EXECUTION
# ============================================

def demo_single_image(model, image_path, expected_label=None):
    """Demo on a single image"""
    print("\n\n" + "=" * 60)
    print(f"TESTING IMAGE: {os.path.basename(image_path)}")
    if expected_label:
        print(f"EXPECTED: {expected_label}")
    print("=" * 60)
    
    # Extract eye
    preprocessed, original = extract_eye(image_path)
    
    if preprocessed is None:
        return
    
    # Predict
    pred, label, conf = predict_eye_state(model, preprocessed)
    
    # Check result
    if expected_label:
        correct = (expected_label.upper() == label)
        print(f"\n{'='*60}")
        print(f"RESULT: {'[CORRECT]' if correct else '[WRONG]'}")
        print(f"{'='*60}")

    # Save visualization
    if original is not None:
        output_path = f"results/demo_{os.path.basename(image_path)}"
        os.makedirs('results', exist_ok=True)
        cv2.imwrite(output_path, original)
        print(f"\n[SAVED] Visualization: {output_path}")


def main():
    """Main demo execution"""
    print("=" * 60)
    print("DEMO: extract_eye() Function Test")
    print("=" * 60)
    
    # Check model
    if not os.path.exists(MODEL_PATH):
        print(f"\n[ERROR] Model not found: {MODEL_PATH}")
        print("   Please ensure eye_state_classifier.h5 is in models/ directory")
        return

    # Load model
    print(f"\n[LOADING] Loading model...")
    try:
        # Use custom loader to handle TF 2.15+ compatibility
        model = load_model_fixed(MODEL_PATH)
        print(f"   [OK] Model loaded from: {MODEL_PATH}")
        print(f"   Input shape: {model.input_shape}")
    except Exception as e:
        print(f"   [ERROR] {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test samples
    samples = [
        (SAMPLE_CLOSED_EYE, "CLOSED"),
        (SAMPLE_OPEN_EYE, "OPEN")
    ]
    
    for img_path, expected in samples:
        if os.path.exists(img_path):
            demo_single_image(model, img_path, expected)
        else:
            print(f"\n[WARNING] Sample not found: {img_path}")
            print("   Update the image paths in the script to match your data")

    print("\n\n" + "=" * 60)
    print("[COMPLETE] DEMO COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Verify the output makes sense")
    print("2. Run test_eye_extraction.py for batch testing")
    print("3. Run main_v2.py for real-time detection")
    print("=" * 60)


if __name__ == "__main__":
    main()
