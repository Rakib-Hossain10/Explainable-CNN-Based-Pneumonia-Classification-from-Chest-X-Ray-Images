# model_utils.py

import os
import json
from pathlib import Path
from typing import Any, Dict, Tuple, Union, List

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

# Keep these filenames if your 3-class best model is saved with these names.
# If your files have different names, change only these two lines.
DEFAULT_MODEL_PATH = BASE_DIR / "Pneumonia_resnet50.keras"
DEFAULT_METADATA_PATH = BASE_DIR / "Pneumonia_resnet50_metadata.json"


# ---------------------------------------------------------
# Expected 3-class setup
# ---------------------------------------------------------

EXPECTED_CLASS_NAMES = [
    "Normal",
    "Pneumonia-Bacterial",
    "Pneumonia-Viral",
]

NORMAL_CLASS_NAME = "Normal"


# ---------------------------------------------------------
# Custom function used inside your trained model
# ---------------------------------------------------------
# Your ResNet training notebook used this function inside a Lambda layer.
# So we must define it again before loading the .keras model.

def crop_borders_tf(images: tf.Tensor) -> tf.Tensor:
    """
    Crop image borders using central crop.

    This must match the function used during training.
    Your previous ResNet model used central_fraction=0.84.
    """
    return tf.image.central_crop(images, central_fraction=0.84)


def _enable_unsafe_lambda_loading() -> None:
    """
    Some Keras versions block Lambda layer deserialization by default.
    This enables loading your own trusted local model file.

    Use this only for your own trained model.
    """
    try:
        tf.keras.config.enable_unsafe_deserialization()
    except Exception:
        try:
            import keras
            keras.config.enable_unsafe_deserialization()
        except Exception:
            pass


# ---------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------

def _normalize_class_name(name: str) -> str:
    """
    Normalize class names for internal comparison only.
    """
    return str(name).strip().lower().replace("_", "-").replace(" ", "-")


def _is_normal_class(class_name: str) -> bool:
    """
    Return True if class name represents Normal.
    """
    return _normalize_class_name(class_name) == "normal"


def _is_pneumonia_class(class_name: str) -> bool:
    """
    Return True if class name represents any pneumonia class.

    For 3-class setup:
    - Pneumonia-Bacterial
    - Pneumonia-Viral
    """
    normalized = _normalize_class_name(class_name)
    return "pneumonia" in normalized and normalized != "normal"


def validate_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clean metadata for the 3-class deployment app.

    Required keys:
    - IMAGE_SIZE
    - CHANNELS
    - class_names

    Expected class_names:
    ["Normal", "Pneumonia-Bacterial", "Pneumonia-Viral"]
    """
    required_keys = ["IMAGE_SIZE", "CHANNELS", "class_names"]
    missing_keys = [key for key in required_keys if key not in metadata]

    if missing_keys:
        raise ValueError(
            f"Metadata file is missing required keys: {missing_keys}"
        )

    metadata["IMAGE_SIZE"] = int(metadata["IMAGE_SIZE"])
    metadata["CHANNELS"] = int(metadata["CHANNELS"])
    metadata["class_names"] = [str(name).strip() for name in metadata["class_names"]]

    class_names = metadata["class_names"]

    if metadata["IMAGE_SIZE"] <= 0:
        raise ValueError("IMAGE_SIZE must be a positive integer.")

    if metadata["CHANNELS"] != 3:
        raise ValueError(
            f"This deployment expects RGB images with 3 channels. "
            f"Metadata CHANNELS={metadata['CHANNELS']}."
        )

    if len(class_names) != 3:
        raise ValueError(
            "This 3-class deployment expects exactly 3 classes: "
            "Normal, Pneumonia-Bacterial, Pneumonia-Viral. "
            f"Found class_names={class_names}"
        )

    normalized_found = [_normalize_class_name(name) for name in class_names]
    normalized_expected = [_normalize_class_name(name) for name in EXPECTED_CLASS_NAMES]

    if normalized_found != normalized_expected:
        raise ValueError(
            "Class names or class order do not match the expected 3-class setup.\n"
            f"Expected order: {EXPECTED_CLASS_NAMES}\n"
            f"Found order: {class_names}\n\n"
            "Important: the class order in metadata must match the model output order."
        )

    metadata["NUM_CLASSES"] = len(class_names)
    metadata["TASK_TYPE"] = "Multi-class Classification"

    return metadata


def load_metadata(
    metadata_path: Union[str, Path] = DEFAULT_METADATA_PATH
) -> Dict[str, Any]:
    """
    Load model metadata JSON.

    Expected metadata example:
    {
        "IMAGE_SIZE": 224,
        "CHANNELS": 3,
        "class_names": ["Normal", "Pneumonia-Bacterial", "Pneumonia-Viral"]
    }
    """
    metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}\n"
            "Make sure Pneumonia_resnet50_metadata.json is inside the deployment folder."
        )

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    metadata = validate_metadata(metadata)

    return metadata


# ---------------------------------------------------------
# Model loading
# ---------------------------------------------------------

def load_pneumonia_model(
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH
) -> tf.keras.Model:
    """
    Load the trained 3-class pneumonia ResNet50 model.

    Important:
    - compile=False is used because we only need inference.
    - custom_objects is needed because your model may use crop_borders_tf.
    - safe_mode=False is needed for Lambda layers in newer Keras versions.
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Make sure Pneumonia_resnet50.keras is inside the deployment folder."
        )

    _enable_unsafe_lambda_loading()

    custom_objects = {
        "crop_borders_tf": crop_borders_tf,
        "tf": tf,
    }

    try:
        model = tf.keras.models.load_model(
            str(model_path),
            custom_objects=custom_objects,
            compile=False,
            safe_mode=False,
        )
    except TypeError:
        # TensorFlow/Keras 2.10 may not support safe_mode.
        model = tf.keras.models.load_model(
            str(model_path),
            custom_objects=custom_objects,
            compile=False,
        )

    return model


def validate_model_output(
    model: tf.keras.Model,
    metadata: Dict[str, Any],
) -> None:
    """
    Validate that the model output matches the metadata class count.
    """
    class_names = metadata["class_names"]
    expected_classes = len(class_names)

    output_shape = model.output_shape

    if isinstance(output_shape, list):
        output_shape = output_shape[0]

    if output_shape[-1] != expected_classes:
        raise ValueError(
            f"Model output shape {model.output_shape} does not match "
            f"metadata class count {expected_classes}.\n"
            f"class_names={class_names}"
        )


def warmup_model(
    model: tf.keras.Model,
    image_size: int,
    channels: int
) -> None:
    """
    Run one dummy prediction so the first real prediction is faster.
    """
    dummy_input = np.zeros(
        (1, image_size, image_size, channels),
        dtype=np.float32
    )

    _ = model(dummy_input, training=False)


def load_model_and_metadata(
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    metadata_path: Union[str, Path] = DEFAULT_METADATA_PATH,
    warmup: bool = True,
) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    """
    Load model and metadata together.

    This function will be cached in Streamlit using st.cache_resource.
    """
    metadata = load_metadata(metadata_path)
    model = load_pneumonia_model(model_path)

    validate_model_output(model, metadata)

    if warmup:
        warmup_model(
            model=model,
            image_size=metadata["IMAGE_SIZE"],
            channels=metadata["CHANNELS"],
        )

    return model, metadata


# ---------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------

def load_image_as_rgb(image_input: Any) -> Image.Image:
    """
    Convert uploaded image, file path, or PIL image into RGB PIL image.

    Supports:
    - Streamlit uploaded file
    - image path
    - PIL Image
    """
    if isinstance(image_input, Image.Image):
        image = image_input.copy()
    else:
        if hasattr(image_input, "seek"):
            try:
                image_input.seek(0)
            except Exception:
                pass

        image = Image.open(image_input)

    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    return image


def preprocess_image_for_prediction(
    image_input: Any,
    image_size: int = 224
) -> Tuple[np.ndarray, Image.Image]:
    """
    Prepare image for model prediction.

    Important:
    Do NOT divide by 255 here if your trained ResNet50 model already contains:
    - crop layer
    - augmentation layer
    - ResNet preprocess_input layer

    This matches your old 2-class deployment logic.
    The input remains in normal 0-255 pixel range.
    """
    image = load_image_as_rgb(image_input)
    image = image.resize((image_size, image_size))

    image_array = np.array(image).astype(np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    # Display version: grayscale-looking RGB image for clean X-ray UI.
    gray_image = image.convert("L")
    display_image = Image.merge("RGB", (gray_image, gray_image, gray_image))

    return image_array, display_image


# ---------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------

def normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """
    Clean and normalize softmax probabilities.

    This makes the app robust even if tiny floating-point issues occur.
    """
    probabilities = np.asarray(probabilities, dtype=np.float32).reshape(-1)

    probabilities = np.nan_to_num(
        probabilities,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    probabilities = np.clip(probabilities, 0.0, 1.0)

    total = float(np.sum(probabilities))

    if total > 0:
        probabilities = probabilities / total

    return probabilities


def get_pneumonia_probability(
    probabilities: np.ndarray,
    class_names: List[str],
) -> float:
    """
    Return total probability of all pneumonia classes.

    For 3-class setup:
    Pneumonia probability = P(Pneumonia-Bacterial) + P(Pneumonia-Viral)
    """
    total_probability = 0.0

    for index, class_name in enumerate(class_names):
        if _is_pneumonia_class(class_name):
            total_probability += float(probabilities[index])

    return float(total_probability)


def get_diagnosis_group(predicted_class: str) -> str:
    """
    Return high-level diagnosis group for UI logic.
    """
    if _is_normal_class(predicted_class):
        return "Normal"

    if _is_pneumonia_class(predicted_class):
        return "Pneumonia"

    return "Unknown"


def get_pneumonia_type(predicted_class: str) -> str:
    """
    Return pneumonia subtype for UI display.
    """
    normalized = _normalize_class_name(predicted_class)

    if normalized == "pneumonia-bacterial":
        return "Bacterial Pneumonia"

    if normalized == "pneumonia-viral":
        return "Viral Pneumonia"

    if _is_normal_class(predicted_class):
        return "Not applicable"

    return predicted_class


def predict_image(
    model: tf.keras.Model,
    image_input: Any,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Predict one of three classes from a chest X-ray image.

    Classes:
    - Normal
    - Pneumonia-Bacterial
    - Pneumonia-Viral

    Returns a clean dictionary for Streamlit UI.
    """
    image_size = int(metadata["IMAGE_SIZE"])
    class_names = list(metadata["class_names"])
    num_classes = len(class_names)

    image_array, display_image = preprocess_image_for_prediction(
        image_input=image_input,
        image_size=image_size,
    )

    predictions = model(image_array, training=False)
    probabilities = normalize_probabilities(predictions)

    if len(probabilities) != num_classes:
        raise ValueError(
            f"Model output size {len(probabilities)} does not match "
            f"class_names size {num_classes}.\n"
            f"class_names={class_names}"
        )

    predicted_index = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])

    pneumonia_probability = get_pneumonia_probability(
        probabilities=probabilities,
        class_names=class_names,
    )

    diagnosis_group = get_diagnosis_group(predicted_class)
    pneumonia_type = get_pneumonia_type(predicted_class)
    pneumonia_detected = diagnosis_group == "Pneumonia"

    probability_dict = {
        class_names[i]: float(probabilities[i])
        for i in range(num_classes)
    }

    probability_percent_dict = {
        class_names[i]: round(float(probabilities[i]) * 100.0, 2)
        for i in range(num_classes)
    }

    result = {
        "predicted_index": predicted_index,
        "predicted_class": predicted_class,
        "diagnosis_group": diagnosis_group,
        "pneumonia_type": pneumonia_type,

        "confidence": confidence,
        "confidence_percent": round(confidence * 100.0, 2),

        "pneumonia_probability": pneumonia_probability,
        "pneumonia_probability_percent": round(pneumonia_probability * 100.0, 2),
        "pneumonia_detected": pneumonia_detected,

        "probabilities": probability_dict,
        "probabilities_percent": probability_percent_dict,

        "display_image": display_image,
        "image_array": image_array,

        "class_names": class_names,
        "task_type": "Multi-class Classification",
    }

    return result