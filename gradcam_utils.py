# gradcam_utils.py

from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter
import matplotlib.cm as cm

try:
    from model_utils import preprocess_image_for_prediction
except ImportError:
    from .model_utils import preprocess_image_for_prediction


# ---------------------------------------------------------
# Default Grad-CAM settings
# ---------------------------------------------------------
# For ResNet50, "conv5_block3_out" is usually the final convolutional layer.
# If your saved model uses a different layer, the code below also tries
# several fallback layer names automatically.

DEFAULT_LAST_CONV_LAYER_NAME = "conv5_block3_out"

DEFAULT_TARGET_LAYER_CANDIDATES = [
    "conv5_block3_out",   # ResNet50 final conv block output
    "conv4_block6_out",   # earlier ResNet50 block, useful if preferred
    "out_relu",           # MobileNet/EfficientNet style
    "top_activation",     # EfficientNet style
    "block7a_project_bn", # EfficientNetB0 style
]


# ---------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------

def _get_resample_method():
    """
    Pillow compatibility helper.
    """
    if hasattr(Image, "Resampling"):
        return Image.Resampling.BILINEAR
    return Image.BILINEAR


def _normalize_class_name(name: str) -> str:
    """
    Normalize class names for comparison.

    Examples:
    "Pneumonia-Bacterial" -> "pneumonia-bacterial"
    "Pneumonia Viral" -> "pneumonia-viral"
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
    """
    normalized = _normalize_class_name(class_name)
    return "pneumonia" in normalized and normalized != "normal"


def _get_diagnosis_group(class_name: str) -> str:
    """
    Return high-level diagnosis group.
    """
    if _is_normal_class(class_name):
        return "Normal"

    if _is_pneumonia_class(class_name):
        return "Pneumonia"

    return "Unknown"


def _get_pneumonia_type(class_name: str) -> str:
    """
    Return readable pneumonia subtype.
    """
    normalized = _normalize_class_name(class_name)

    if normalized == "pneumonia-bacterial":
        return "Bacterial Pneumonia"

    if normalized == "pneumonia-viral":
        return "Viral Pneumonia"

    if _is_normal_class(class_name):
        return "Not applicable"

    return str(class_name)


def apply_layer_inference(layer: tf.keras.layers.Layer, x: tf.Tensor) -> tf.Tensor:
    """
    Apply a layer in inference mode.

    Some layers accept training=False.
    Some layers do not.
    This helper handles both cases safely.
    """
    try:
        return layer(x, training=False)
    except TypeError:
        return layer(x)


def _validate_class_index(class_index: int, class_names: List[str]) -> int:
    """
    Validate class index.
    """
    class_index = int(class_index)

    if class_index < 0 or class_index >= len(class_names):
        raise ValueError(
            f"class_index={class_index} is invalid. "
            f"Valid range is 0 to {len(class_names) - 1}."
        )

    return class_index


def resolve_target_class_index(
    class_names: List[str],
    prediction_vector: Optional[np.ndarray] = None,
    class_index: Optional[int] = None,
    target_class_name: Optional[str] = None,
) -> int:
    """
    Decide which class Grad-CAM should explain.

    Priority:
    1. If class_index is given, use it.
    2. Else if target_class_name is given, use that class.
    3. Else use predicted class from prediction_vector.

    For your 3-class project:
    - Normal
    - Pneumonia-Bacterial
    - Pneumonia-Viral
    """
    if class_index is not None:
        return _validate_class_index(class_index, class_names)

    if target_class_name is not None:
        normalized_names = [_normalize_class_name(name) for name in class_names]
        normalized_target = _normalize_class_name(target_class_name)

        if normalized_target not in normalized_names:
            raise ValueError(
                f"target_class_name='{target_class_name}' was not found in "
                f"class_names={class_names}.\n"
                "For your 3-class model, use one of: "
                "'Normal', 'Pneumonia-Bacterial', or 'Pneumonia-Viral'."
            )

        return int(normalized_names.index(normalized_target))

    if prediction_vector is None:
        raise ValueError(
            "prediction_vector is required when both class_index and "
            "target_class_name are None."
        )

    return int(np.argmax(prediction_vector))


def normalize_prediction_vector(predictions: Any) -> np.ndarray:
    """
    Convert model prediction output into a clean probability vector.
    """
    prediction_vector = np.asarray(predictions, dtype=np.float32).reshape(-1)

    prediction_vector = np.nan_to_num(
        prediction_vector,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    prediction_vector = np.clip(prediction_vector, 0.0, 1.0)

    total = float(np.sum(prediction_vector))

    if total > 0:
        prediction_vector = prediction_vector / total

    return prediction_vector


# ---------------------------------------------------------
# Target convolution layer discovery
# ---------------------------------------------------------

def _try_get_layer(model: tf.keras.Model, layer_name: str):
    """
    Try to get a layer by name. Return None if not found.
    """
    try:
        return model.get_layer(layer_name)
    except Exception:
        return None


def find_nested_model_with_target_layer(
    full_model: tf.keras.Model,
    target_layer_name: str,
) -> Tuple[tf.keras.Model, int]:
    """
    Find a nested model/backbone inside the full model that contains target_layer_name.

    This is useful when the trained model is like:

    Input
    -> crop layer
    -> data augmentation
    -> ResNet preprocess_input Lambda
    -> ResNet50 backbone
    -> GlobalAveragePooling2D
    -> Dense
    -> Dense softmax
    """
    for index, layer in enumerate(full_model.layers):
        if hasattr(layer, "get_layer"):
            try:
                _ = layer.get_layer(target_layer_name)
                return layer, index
            except Exception:
                pass

    raise ValueError(
        f"Could not find nested model containing layer '{target_layer_name}'."
    )


def find_available_target_layer_name(
    full_model: tf.keras.Model,
    preferred_layer_name: Optional[str] = None,
    candidate_layer_names: Optional[List[str]] = None,
) -> str:
    """
    Find the best available target convolution layer name.

    Search order:
    1. preferred_layer_name if provided
    2. default candidate names
    3. any top-level Conv2D layer from the end of the model
    4. any nested Conv2D layer from the end of nested models
    """
    candidates = []

    if preferred_layer_name:
        candidates.append(preferred_layer_name)

    if candidate_layer_names:
        candidates.extend(candidate_layer_names)
    else:
        candidates.extend(DEFAULT_TARGET_LAYER_CANDIDATES)

    # Remove duplicates while preserving order.
    seen = set()
    candidates = [
        name for name in candidates
        if not (name in seen or seen.add(name))
    ]

    # Check preferred/candidate layer names in nested models and direct model.
    for layer_name in candidates:
        # Direct top-level check.
        if _try_get_layer(full_model, layer_name) is not None:
            return layer_name

        # Nested model check.
        for layer in full_model.layers:
            if hasattr(layer, "get_layer"):
                try:
                    _ = layer.get_layer(layer_name)
                    return layer_name
                except Exception:
                    pass

    # Fallback 1: find last top-level Conv2D layer.
    for layer in reversed(full_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name

    # Fallback 2: find last Conv2D layer inside nested models.
    for layer in reversed(full_model.layers):
        if hasattr(layer, "layers"):
            for nested_layer in reversed(layer.layers):
                if isinstance(nested_layer, tf.keras.layers.Conv2D):
                    return nested_layer.name

    raise ValueError(
        "Could not automatically find a convolutional layer for Grad-CAM. "
        "Please pass target_layer_name manually."
    )


# ---------------------------------------------------------
# Build Grad-CAM model
# ---------------------------------------------------------

def build_gradcam_model(
    full_model: tf.keras.Model,
    target_layer_name: Optional[str] = None,
) -> tf.keras.Model:
    """
    Build a Grad-CAM model.

    This model returns:
    1. Output of the selected convolution layer
    2. Final model prediction

    Output:
        conv_outputs, predictions

    For your ResNet50 model, the recommended target layer is:
        conv5_block3_out
    """
    selected_layer_name = find_available_target_layer_name(
        full_model=full_model,
        preferred_layer_name=target_layer_name or DEFAULT_LAST_CONV_LAYER_NAME,
    )

    # First try nested-backbone approach.
    try:
        base_model, base_index = find_nested_model_with_target_layer(
            full_model=full_model,
            target_layer_name=selected_layer_name,
        )

        target_layer = base_model.get_layer(selected_layer_name)

        feature_extractor = tf.keras.Model(
            inputs=base_model.input,
            outputs=[target_layer.output, base_model.output],
            name="feature_extractor_for_gradcam",
        )

        input_shape = full_model.input_shape

        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        model_input = tf.keras.Input(shape=input_shape[1:])
        x = model_input

        # Layers before backbone.
        for layer in full_model.layers[:base_index]:
            x = apply_layer_inference(layer, x)

        # Backbone until target conv layer.
        conv_outputs, x = feature_extractor(x, training=False)

        # Layers after backbone.
        for layer in full_model.layers[base_index + 1:]:
            x = apply_layer_inference(layer, x)

        predictions = x

        gradcam_model = tf.keras.Model(
            inputs=model_input,
            outputs=[conv_outputs, predictions],
            name="pneumonia_3class_gradcam_model",
        )

        gradcam_model.selected_target_layer_name = selected_layer_name

        return gradcam_model

    except Exception as nested_error:
        # Fallback for models where target layer is directly available.
        try:
            target_layer = full_model.get_layer(selected_layer_name)

            gradcam_model = tf.keras.Model(
                inputs=full_model.input,
                outputs=[target_layer.output, full_model.output],
                name="pneumonia_3class_gradcam_model",
            )

            gradcam_model.selected_target_layer_name = selected_layer_name

            return gradcam_model

        except Exception as direct_error:
            raise ValueError(
                "Could not build Grad-CAM model.\n"
                f"Selected target layer: {selected_layer_name}\n"
                f"Nested model error: {nested_error}\n"
                f"Direct model error: {direct_error}"
            )


# ---------------------------------------------------------
# Create Grad-CAM heatmap
# ---------------------------------------------------------

def make_gradcam_heatmap(
    image_array: np.ndarray,
    gradcam_model: tf.keras.Model,
    class_index: Optional[int] = None,
    image_size: int = 224,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Create raw Grad-CAM heatmap.

    Args:
        image_array:
            Preprocessed image array with shape:
            (1, image_size, image_size, 3)

        gradcam_model:
            Model created by build_gradcam_model()

        class_index:
            Which class to explain.
            If None, explains the predicted class.

        image_size:
            Final heatmap resize size.

    Returns:
        heatmap:
            2D numpy array in range [0, 1]

        predictions:
            Model prediction probabilities

        used_class_index:
            Class index used for Grad-CAM
    """
    if image_array.ndim != 4:
        raise ValueError(
            f"image_array must have shape (1, H, W, 3). "
            f"Received shape: {image_array.shape}"
        )

    image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = gradcam_model(image_tensor, training=False)

        prediction_vector = normalize_prediction_vector(predictions)

        if class_index is None:
            used_class_index = int(np.argmax(prediction_vector))
        else:
            used_class_index = int(class_index)

        class_channel = predictions[:, used_class_index]

    gradients = tape.gradient(class_channel, conv_outputs)

    if gradients is None:
        raise ValueError(
            "Gradients are None. Grad-CAM could not be generated. "
            "Please check that the target layer is connected to the model output."
        )

    # Average gradients over height and width.
    pooled_gradients = tf.reduce_mean(gradients, axis=(1, 2))

    # Remove batch dimension.
    conv_outputs = conv_outputs[0]
    pooled_gradients = pooled_gradients[0]

    # Weighted sum of convolution feature maps.
    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients[tf.newaxis, tf.newaxis, :],
        axis=-1,
    )

    # ReLU: keep only positive influence.
    heatmap = tf.maximum(heatmap, 0)

    # Normalize to [0, 1].
    max_value = tf.reduce_max(heatmap)
    heatmap = heatmap / (max_value + 1e-8)

    # Resize heatmap to image size.
    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis],
        (image_size, image_size),
        method="bilinear",
    )

    heatmap = tf.squeeze(heatmap).numpy()
    heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=1.0, neginf=0.0)
    heatmap = np.clip(heatmap, 0.0, 1.0)

    return heatmap, normalize_prediction_vector(predictions), used_class_index


# ---------------------------------------------------------
# Overlay Grad-CAM on X-ray
# ---------------------------------------------------------

def overlay_gradcam_on_xray(
    display_img: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.55,
    cmap_name: str = "jet",
    blur_radius: int = 1,
    min_percentile: Optional[int] = 98,
    activation_cutoff: float = 0.45,
) -> Tuple[Image.Image, Image.Image]:
    """
    Overlay Grad-CAM heatmap on X-ray image.

    This version hides weak green/cyan regions and only shows stronger
    Grad-CAM activations. This gives a cleaner visualization for thesis
    screenshots and deployment demonstration.
    """
    if not isinstance(display_img, Image.Image):
        raise TypeError("display_img must be a PIL Image.")

    display_img = display_img.convert("RGB")

    heatmap = np.asarray(heatmap, dtype=np.float32)
    heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=1.0, neginf=0.0)
    heatmap = np.maximum(heatmap, 0.0)

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    # Step 1: keep only top Grad-CAM activation regions.
    if min_percentile is not None:
        threshold = np.percentile(heatmap, min_percentile)
        mask = np.where(heatmap >= threshold, heatmap, 0.0)
    else:
        mask = heatmap.copy()

    if mask.max() > 0:
        mask = mask / mask.max()

    # Step 2: remove weak visible regions.
    mask = np.where(mask >= activation_cutoff, mask, 0.0)

    if mask.max() > 0:
        mask = mask / mask.max()

    mask = np.power(mask, 1.8)
    mask = np.clip(mask, 0.0, 1.0)

    cmap = cm.get_cmap(cmap_name)

    colored_heatmap = cmap(mask)[:, :, :3]
    colored_heatmap = np.uint8(colored_heatmap * 255)

    heatmap_color_img = Image.fromarray(colored_heatmap).resize(
        display_img.size,
        _get_resample_method(),
    )

    mask_img = Image.fromarray(np.uint8(mask * 255)).resize(
        display_img.size,
        _get_resample_method(),
    )

    if blur_radius and blur_radius > 0:
        heatmap_color_img = heatmap_color_img.filter(
            ImageFilter.GaussianBlur(radius=blur_radius)
        )
        mask_img = mask_img.filter(
            ImageFilter.GaussianBlur(radius=blur_radius)
        )

    # Step 3: after blur, weak pixels can reappear. Remove them again.
    mask_array = np.asarray(mask_img).astype(np.float32) / 255.0
    mask_array = np.where(mask_array >= activation_cutoff, mask_array, 0.0)

    if mask_array.max() > 0:
        mask_array = mask_array / mask_array.max()

    alpha_mask_array = np.uint8(mask_array * float(alpha) * 255)
    alpha_mask_img = Image.fromarray(alpha_mask_array, mode="L")

    overlay_img = Image.composite(
        heatmap_color_img,
        display_img,
        alpha_mask_img,
    )

    black_background = Image.new("RGB", display_img.size, color=(0, 0, 0))
    heatmap_img = Image.composite(
        heatmap_color_img,
        black_background,
        alpha_mask_img,
    )

    return overlay_img, heatmap_img


# ---------------------------------------------------------
# Main Grad-CAM function for Streamlit
# ---------------------------------------------------------

def generate_gradcam(
    gradcam_model: tf.keras.Model,
    image_input: Any,
    metadata: Dict[str, Any],
    class_index: Optional[int] = None,
    target_class_name: Optional[str] = None,
    alpha: float = 0.55,
    cmap_name: str = "jet",
    blur_radius: int = 1,
    min_percentile: Optional[int] = 98,
    activation_cutoff: float = 0.45,
) -> Dict[str, Any]:
    """
    Generate Grad-CAM result for Streamlit.

    For your 3-class project:
    class_names should be:
        ["Normal", "Pneumonia-Bacterial", "Pneumonia-Viral"]

    If target_class_name is None and class_index is None,
    Grad-CAM explains the predicted class automatically.

    Example 1: explain predicted class
        result = generate_gradcam(
            gradcam_model=gradcam_model,
            image_input=uploaded_image,
            metadata=metadata,
        )

    Example 2: explain bacterial pneumonia class
        result = generate_gradcam(
            gradcam_model=gradcam_model,
            image_input=uploaded_image,
            metadata=metadata,
            target_class_name="Pneumonia-Bacterial"
        )

    Returns dictionary containing:
    - display_img
    - overlay_img
    - heatmap_img
    - heatmap
    - prediction details
    - Grad-CAM target details
    """
    image_size = int(metadata["IMAGE_SIZE"])
    class_names = list(metadata["class_names"])

    image_array, display_img = preprocess_image_for_prediction(
        image_input=image_input,
        image_size=image_size,
    )

    # First pass: get predictions using predicted class by default if needed.
    _, raw_predictions, _ = make_gradcam_heatmap(
        image_array=image_array,
        gradcam_model=gradcam_model,
        class_index=None,
        image_size=image_size,
    )

    prediction_vector = normalize_prediction_vector(raw_predictions)

    if len(prediction_vector) != len(class_names):
        raise ValueError(
            f"Model output size {len(prediction_vector)} does not match "
            f"class_names size {len(class_names)}."
        )

    resolved_class_index = resolve_target_class_index(
        class_names=class_names,
        prediction_vector=prediction_vector,
        class_index=class_index,
        target_class_name=target_class_name,
    )

    heatmap, predictions, used_class_index = make_gradcam_heatmap(
        image_array=image_array,
        gradcam_model=gradcam_model,
        class_index=resolved_class_index,
        image_size=image_size,
    )

    prediction_vector = normalize_prediction_vector(predictions)

    predicted_index = int(np.argmax(prediction_vector))
    predicted_class = class_names[predicted_index]
    confidence = float(prediction_vector[predicted_index])

    used_class_name = class_names[used_class_index]
    used_class_probability = float(prediction_vector[used_class_index])

    probabilities = {
        class_names[i]: float(prediction_vector[i])
        for i in range(len(class_names))
    }

    probabilities_percent = {
        class_names[i]: round(float(prediction_vector[i]) * 100.0, 2)
        for i in range(len(class_names))
    }

    pneumonia_probability = 0.0
    for i, class_name in enumerate(class_names):
        if _is_pneumonia_class(class_name):
            pneumonia_probability += float(prediction_vector[i])

    diagnosis_group = _get_diagnosis_group(predicted_class)
    pneumonia_type = _get_pneumonia_type(predicted_class)
    pneumonia_detected = diagnosis_group == "Pneumonia"

    overlay_img, heatmap_img = overlay_gradcam_on_xray(
        display_img=display_img,
        heatmap=heatmap,
        alpha=alpha,
        cmap_name=cmap_name,
        blur_radius=blur_radius,
        min_percentile=min_percentile,
        activation_cutoff=activation_cutoff,
    )

    selected_target_layer_name = getattr(
        gradcam_model,
        "selected_target_layer_name",
        "unknown",
    )

    result = {
        "display_img": display_img,
        "overlay_img": overlay_img,
        "heatmap_img": heatmap_img,
        "heatmap": heatmap,

        "predictions": prediction_vector,
        "predicted_index": predicted_index,
        "predicted_class": predicted_class,
        "diagnosis_group": diagnosis_group,
        "pneumonia_type": pneumonia_type,

        "confidence": confidence,
        "confidence_percent": round(confidence * 100.0, 2),

        "probabilities": probabilities,
        "probabilities_percent": probabilities_percent,

        "pneumonia_probability": float(pneumonia_probability),
        "pneumonia_probability_percent": round(float(pneumonia_probability) * 100.0, 2),
        "pneumonia_detected": pneumonia_detected,

        "used_class_index": int(used_class_index),
        "used_class_name": used_class_name,
        "used_class_probability": used_class_probability,
        "used_class_probability_percent": round(used_class_probability * 100.0, 2),
        "target_class_name": used_class_name,

        "target_layer_name": selected_target_layer_name,
        "task_type": "Multi-class Classification",
    }

    return result


# ---------------------------------------------------------
# Optional helper: create side-by-side image
# ---------------------------------------------------------

def create_side_by_side_gradcam_image(
    display_img: Image.Image,
    overlay_img: Image.Image,
) -> Image.Image:
    """
    Create a single side-by-side image:
    original X-ray | Grad-CAM overlay

    Useful for downloadable result image.
    """
    display_img = display_img.convert("RGB")
    overlay_img = overlay_img.convert("RGB")

    width, height = display_img.size

    side_by_side = Image.new(
        "RGB",
        (width * 2, height),
        color=(255, 255, 255),
    )

    side_by_side.paste(display_img, (0, 0))
    side_by_side.paste(overlay_img, (width, 0))

    return side_by_side