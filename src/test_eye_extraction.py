"""
Test Script for Eye Extraction and CNN Prediction
Tests the extract_eye() function and model predictions on sample images
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt

# ============================================
# CONFIGURATION
# ============================================
MODEL_PATH = "C:\\Users\\hp\\Desktop\\Real_Time_Driver_Drowsiness_Detection\\models\\eye_state_classifier.h5"
EYE_IMG_SIZE = (64, 64)

# Test with sample data
TEST_DATA_DIR = "C:\\Users\\hp\\Desktop\\Real_Time_Driver_Drowsiness_Detection\\data\\eyes\\test"
SAMPLE_CLASSES = ["Close", "Open"]
SAMPLES_PER_CLASS = 3  # Changed to 3 images per class


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
# EYE EXTRACTION FUNCTION (same as main.py)
# ============================================

def extract_eye(eye_image_path, resize_to=(64, 64)):
    """
    Extract and preprocess eye image for CNN model
    
    Args:
        eye_image_path: Path to eye image file
        resize_to: Target size (width, height)
    
    Returns:
        Preprocessed eye image ready for model prediction
        Shape: (1, 64, 64, 1)
    """
    # Read image
    img = cv2.imread(eye_image_path)

    if img is None:
        print(f"[ERROR] Could not read image: {eye_image_path}")
        return None
    
    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize to model input size
    img_resized = cv2.resize(img_gray, resize_to)
    
    # Normalize to [0, 1] (same as training)
    img_normalized = img_resized / 255.0
    
    # Add channel dimension: (64, 64) -> (64, 64, 1)
    img_normalized = np.expand_dims(img_normalized, axis=-1)
    
    # Add batch dimension: (64, 64, 1) -> (1, 64, 64, 1)
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch, img_gray


def predict_eye_state(model, eye_image):
    """
    Predict if eye is open or closed
    
    Args:
        model: Loaded Keras model
        eye_image: Preprocessed eye image (1, 64, 64, 1)
    
    Returns:
        prediction: Float between 0-1
        is_closed: Boolean
        confidence: Percentage
        label: String "OPEN" or "CLOSED"
    """
    if eye_image is None:
        return None, None, None, None
    
    # Get prediction
    prediction = model.predict(eye_image, verbose=0)[0][0]
    
    # Class 0 = Closed, Class 1 = Open
    is_closed = prediction < 0.5
    label = "CLOSED" if is_closed else "OPEN"
    
    # Calculate confidence
    confidence = (1 - prediction) * 100 if is_closed else prediction * 100
    
    return prediction, is_closed, confidence, label


# ============================================
# TEST FUNCTIONS
# ============================================

def test_single_image(model, image_path, expected_class=None):
    """Test prediction on a single image"""
    print(f"\n{'='*60}")
    print(f"Testing: {os.path.basename(image_path)}")
    if expected_class:
        print(f"Expected: {expected_class}")
    print(f"{'='*60}")

    # Extract and preprocess
    eye_batch, eye_gray = extract_eye(image_path)

    if eye_batch is None:
        print("[ERROR] Failed to extract eye image")
        return

    # Predict
    pred, is_closed, conf, label = predict_eye_state(model, eye_batch)

    # Display results
    print(f"Prediction: {label}")
    print(f"Confidence: {conf:.2f}%")
    print(f"Raw output: {pred:.4f}")

    # Check if correct
    if expected_class:
        # Handle both "Close" and "CLOSED" / "Open" and "OPEN"
        expected_normalized = "CLOSED" if "close" in expected_class.lower() else "OPEN"
        correct = (expected_normalized == label)
        print(f"Result: {'[CORRECT]' if correct else '[WRONG]'}")
        return correct

    return True


def test_batch_samples(model, num_samples=5):
    """Test predictions on multiple samples from each class"""
    print("\n" + "=" * 60)
    print(f"Testing {num_samples} samples from each class")
    print("=" * 60)

    results = {"Close": [], "Open": []}

    for class_name in SAMPLE_CLASSES:
        class_dir = os.path.join(TEST_DATA_DIR, class_name)

        if not os.path.exists(class_dir):
            print(f"\n[WARNING] Directory not found: {class_dir}")
            print("   Make sure you have preprocessed test data")
            continue

        # Get sample images
        images = [f for f in os.listdir(class_dir)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if len(images) == 0:
            print(f"\n[WARNING] No images found in: {class_dir}")
            continue

        # Test samples
        print(f"\n[TESTING] Testing {class_name} samples:")
        print("-" * 60)
        
        for i, img_name in enumerate(images[:num_samples]):
            img_path = os.path.join(class_dir, img_name)
            correct = test_single_image(model, img_path, expected_class=class_name)
            results[class_name].append(correct)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for class_name in SAMPLE_CLASSES:
        if results[class_name]:
            accuracy = (sum(results[class_name]) / len(results[class_name])) * 100
            print(f"{class_name}: {sum(results[class_name])}/{len(results[class_name])} correct ({accuracy:.1f}%)")
    
    # Overall
    all_results = results["Close"] + results["Open"]
    if all_results:
        overall_acc = (sum(all_results) / len(all_results)) * 100
        print(f"\nOverall: {sum(all_results)}/{len(all_results)} correct ({overall_acc:.1f}%)")


def visualize_predictions(model, num_samples=4):
    """Visualize predictions with images"""
    print("\n" + "=" * 60)
    print("Visualizing Predictions")
    print("=" * 60)

    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    fig.suptitle('Eye State Classification Results', fontsize=16)

    for class_idx, class_name in enumerate(SAMPLE_CLASSES):
        class_dir = os.path.join(TEST_DATA_DIR, class_name)

        if not os.path.exists(class_dir):
            print(f"[WARNING] Directory not found: {class_dir}")
            continue
        
        images = [f for f in os.listdir(class_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:num_samples]
        
        for img_idx, img_name in enumerate(images):
            img_path = os.path.join(class_dir, img_name)
            
            # Process image
            eye_batch, eye_gray = extract_eye(img_path)
            
            if eye_batch is None:
                continue
            
            # Predict
            pred, is_closed, conf, label = predict_eye_state(model, eye_batch)
            
            # Plot
            ax = axes[class_idx, img_idx]
            ax.imshow(eye_gray, cmap='gray')
            
            # Color based on correctness
            correct = (class_name.upper() == label)
            color = 'green' if correct else 'red'
            
            ax.set_title(f"True: {class_name}\nPred: {label}\nConf: {conf:.1f}%", 
                        color=color, fontsize=10)
            ax.axis('off')
    
    plt.tight_layout()
    # Save figure to the root 'results' folder (one level up from src)
    root_results_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'eye_classification_test.png'))
    plt.savefig(root_results_path, dpi=150, bbox_inches='tight')
    print("[OK] Visualization saved to: results/eye_classification_test.png")
    plt.show()


# ============================================
# MAIN TEST EXECUTION
# ============================================

def main():
    """Main test execution"""
    print("=" * 60)
    print("Eye State Classification - Test Suite")
    print("=" * 60)

    # Check model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\n[ERROR] Model not found at: {MODEL_PATH}")
        print("Please ensure eye_state_classifier.h5 is in the models/ directory")
        return

    # Load model
    print(f"\n[LOADING] Loading model from: {MODEL_PATH}")
    try:
        model = load_model_fixed(MODEL_PATH)
        print("[OK] Model loaded successfully")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run tests
    print("\n" + "=" * 60)
    print("RUNNING TESTS")
    print("=" * 60)
    
    # Test 1: Batch samples
    test_batch_samples(model, num_samples=SAMPLES_PER_CLASS)
    
    # Test 2: Visualization (optional)
    try:
        os.makedirs('results', exist_ok=True)
        visualize_predictions(model, num_samples=3)
    except Exception as e:
        print(f"\n[WARNING] Visualization skipped: {e}")

    print("\n" + "=" * 60)
    print("[COMPLETE] All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
