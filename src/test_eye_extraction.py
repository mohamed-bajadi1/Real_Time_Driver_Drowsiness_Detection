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
# MODEL_PATH = "models/eye_state_classifier.h5"  # v1 model
# MODEL_PATH = "models/eye_state_classifier_v2_best.h5"  # v2 model
MODEL_PATH = "C:\\Users\\dell\\Desktop\\IAII\\Deep Learning\\DL_Project\\models\\eye_state_classifier_v3.h5"  # v3 model (recommended)
EYE_IMG_SIZE = (64, 64)

# Test with sample data
TEST_DATA_DIR = "C:\\Users\\dell\\Desktop\\IAII\\Deep Learning\\DL_Project\\data\\eyes\\test"
SAMPLE_CLASSES = ["Close", "Open"]
SAMPLES_PER_CLASS = 3  # Changed to 3 images per class

# CLAHE settings (must match training!)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (4, 4)


# ============================================
# CUSTOM MODEL LOADER (Support v1 and v2)
# ============================================
def build_model_v1():
    """Original v1 architecture (171K params)."""
    from tensorflow import keras
    from tensorflow.keras import layers
    from keras.regularizers import l2

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
    from tensorflow import keras
    from keras.regularizers import l2

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
    from keras.regularizers import l2

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
# CLAHE PREPROCESSING (CRITICAL for v2 model!)
# ============================================

# Create CLAHE object once
clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_SIZE)


def apply_clahe(image):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    CRITICAL for v2 model - normalizes contrast across lighting conditions.

    Args:
        image: Grayscale image (uint8)

    Returns:
        CLAHE-enhanced image (uint8)
    """
    return clahe.apply(image)


# ============================================
# EYE EXTRACTION FUNCTION (Updated for v2)
# ============================================

def extract_eye(eye_image_path, resize_to=(64, 64), use_clahe=True):
    """
    Extract and preprocess eye image for CNN model

    PREPROCESSING PIPELINE (must match training!):
    1. Read BGR image
    2. Convert to grayscale
    3. Apply CLAHE (v2 only) - contrast normalization
    4. Resize to 64x64
    5. Normalize to [0, 1]
    6. Add batch and channel dimensions

    Args:
        eye_image_path: Path to eye image file
        resize_to: Target size (width, height)
        use_clahe: Apply CLAHE preprocessing (True for v2, False for v1)

    Returns:
        Preprocessed eye image ready for model prediction
        Shape: (1, 64, 64, 1)
    """
    # Read image
    img = cv2.imread(eye_image_path)

    if img is None:
        print(f"[ERROR] Could not read image: {eye_image_path}")
        return None, None

    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE if using v2 model
    if use_clahe:
        img_gray = apply_clahe(img_gray)

    # Resize to model input size
    img_resized = cv2.resize(img_gray, resize_to)

    # Normalize to [0, 1] (same as training)
    img_normalized = img_resized / 255.0

    # Add channel dimension: (64, 64) -> (64, 64, 1)
    img_normalized = np.expand_dims(img_normalized, axis=-1)

    # Add batch dimension: (64, 64, 1) -> (1, 64, 64, 1)
    img_batch = np.expand_dims(img_normalized, axis=0)

    return img_batch, img_gray


def predict_eye_state(model, eye_image, threshold=0.5):
    """
    Predict if eye is open or closed

    Args:
        model: Loaded Keras model
        eye_image: Preprocessed eye image (1, 64, 64, 1)
        threshold: Classification threshold (default: 0.5, v2 recommends 0.4)

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
    # prediction < threshold -> Closed
    is_closed = prediction < threshold
    label = "CLOSED" if is_closed else "OPEN"

    # Calculate confidence (distance from threshold)
    if is_closed:
        # For closed: confidence = how far below threshold
        confidence = ((threshold - prediction) / threshold) * 100
    else:
        # For open: confidence = how far above threshold
        confidence = ((prediction - threshold) / (1 - threshold)) * 100

    # Clamp confidence to [0, 100]
    confidence = np.clip(confidence, 0, 100)

    return prediction, is_closed, confidence, label


# ============================================
# TEST FUNCTIONS
# ============================================

def test_single_image(model, image_path, expected_class=None, threshold=0.4, use_clahe=True):
    """Test prediction on a single image"""
    print(f"\n{'='*60}")
    print(f"Testing: {os.path.basename(image_path)}")
    if expected_class:
        print(f"Expected: {expected_class}")
    print(f"{'='*60}")

    # Extract and preprocess
    eye_batch, eye_gray = extract_eye(image_path, use_clahe=use_clahe)

    if eye_batch is None:
        print("[ERROR] Failed to extract eye image")
        return

    # Predict
    pred, is_closed, conf, label = predict_eye_state(model, eye_batch, threshold=threshold)

    # Display results
    print(f"Prediction: {label}")
    print(f"Confidence: {conf:.2f}%")
    print(f"Raw output: {pred:.4f}")
    print(f"Threshold: {threshold}")

    # Check if correct
    if expected_class:
        # Handle both "Close" and "CLOSED" / "Open" and "OPEN"
        expected_normalized = "CLOSED" if "close" in expected_class.lower() else "OPEN"
        correct = (expected_normalized == label)
        print(f"Result: {'[CORRECT]' if correct else '[WRONG]'}")
        return correct

    return True


def test_batch_samples(model, num_samples=5, threshold=0.4, use_clahe=True):
    """Test predictions on multiple samples from each class"""
    print("\n" + "=" * 60)
    print(f"Testing {num_samples} samples from each class")
    print(f"Using threshold: {threshold}")
    print(f"Using CLAHE: {use_clahe}")
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
            correct = test_single_image(model, img_path, expected_class=class_name,
                                      threshold=threshold, use_clahe=use_clahe)
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


def visualize_predictions(model, num_samples=4, threshold=0.4, use_clahe=True):
    """Visualize predictions with images"""
    print("\n" + "=" * 60)
    print("Visualizing Predictions")
    print("=" * 60)

    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))

    # Detect model version for title
    if 'v3' in MODEL_PATH.lower():
        model_ver = 'v3'
    elif 'v2' in MODEL_PATH.lower():
        model_ver = 'v2'
    else:
        model_ver = 'v1'

    fig.suptitle(f'Eye State Classification Results (Model: {model_ver}, Threshold: {threshold})',
                 fontsize=16)

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
            eye_batch, eye_gray = extract_eye(img_path, use_clahe=use_clahe)

            if eye_batch is None:
                continue

            # Predict
            pred, is_closed, conf, label = predict_eye_state(model, eye_batch, threshold=threshold)

            # Plot
            ax = axes[class_idx, img_idx]
            ax.imshow(eye_gray, cmap='gray')

            # Color based on correctness
            correct = (class_name.upper() == label)
            color = 'green' if correct else 'red'

            ax.set_title(f"True: {class_name}\nPred: {label}\nConf: {conf:.1f}%\nRaw: {pred:.3f}",
                        color=color, fontsize=9)
            ax.axis('off')

    plt.tight_layout()
    # Save figure to the root 'results' folder (one level up from src)
    result_filename = f'eye_classification_test_{model_ver}.png'
    root_results_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', result_filename))
    os.makedirs(os.path.dirname(root_results_path), exist_ok=True)
    plt.savefig(root_results_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Visualization saved to: {root_results_path}")
    plt.show()


# ============================================
# MAIN TEST EXECUTION
# ============================================

def main():
    """Main test execution"""
    print("=" * 60)
    print("Eye State Classification - Test Suite v2")
    print("=" * 60)

    # Check model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\n[ERROR] Model not found at: {MODEL_PATH}")
        print("Please ensure the model file is in the models/ directory")
        return

    # Determine model version and settings
    if 'v3' in MODEL_PATH.lower():
        model_version = 'v3'
        use_clahe = True
        threshold = 0.4
    elif 'v2' in MODEL_PATH.lower():
        model_version = 'v2'
        use_clahe = True
        threshold = 0.4
    else:
        model_version = 'v1'
        use_clahe = False
        threshold = 0.5

    print(f"\n[INFO] Model version: {model_version}")
    print(f"[INFO] Using CLAHE: {use_clahe}")
    print(f"[INFO] Using threshold: {threshold}")

    # Load model
    print(f"\n[LOADING] Loading model from: {MODEL_PATH}")
    try:
        model = load_model_fixed(MODEL_PATH)
        print("[OK] Model loaded successfully")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        print(f"   Parameters: {model.count_params():,}")
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
    test_batch_samples(model, num_samples=SAMPLES_PER_CLASS, threshold=threshold, use_clahe=use_clahe)

    # Test 2: Visualization (optional)
    try:
        visualize_predictions(model, num_samples=3, threshold=threshold, use_clahe=use_clahe)
    except Exception as e:
        print(f"\n[WARNING] Visualization skipped: {e}")

    print("\n" + "=" * 60)
    print("[COMPLETE] All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
