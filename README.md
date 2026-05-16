# Explainable CNN-Based Pneumonia Classification from Chest X-Ray Images

## Project Overview

This project is a web-based deep learning application for **three-class pneumonia classification** from chest X-ray images. The system classifies an uploaded chest X-ray image into one of the following categories:

1. **Normal**
2. **Pneumonia-Bacterial**
3. **Pneumonia-Viral**

The application uses a trained **ResNet50 convolutional neural network** and integrates **Grad-CAM** visualization to improve the interpretability of model predictions. The system was developed as part of the graduation thesis:

**Explainable CNN-Based Pneumonia Detection from Chest X-Ray Images: A Comparative Study Using Grad-CAM**

The main purpose of this application is to demonstrate how deep learning and explainable artificial intelligence can be combined to support chest X-ray image classification in an academic research environment.

---

## Key Features

- Three-class chest X-ray classification:
  - Normal
  - Pneumonia-Bacterial
  - Pneumonia-Viral

- Deep learning model based on **ResNet50**

- Image upload interface using **Streamlit**

- Automatic preprocessing of uploaded chest X-ray images

- Prediction confidence display

- Class probability visualization for all three classes

- Total pneumonia probability calculation:
  - Pneumonia-Bacterial probability
  - Pneumonia-Viral probability
  - Combined pneumonia probability

- Grad-CAM explainability visualization

- User-selectable Grad-CAM target class

- Side-by-side visualization:
  - Original chest X-ray
  - Grad-CAM overlay

- Downloadable Grad-CAM result image

- Modern dark-blue glass-style user interface suitable for thesis demonstration

---

## System Architecture

The system follows the workflow below:

```text
User Uploads Chest X-ray Image
        ↓
Image Reading and RGB Conversion
        ↓
Image Resizing to 224 × 224
        ↓
ResNet50-Based Classification Model
        ↓
Prediction Result and Class Probabilities
        ↓
Grad-CAM Heatmap Generation
        ↓
Visual Explanation and Result Display



The application consists of three main Python files:

deployment1/
│
├── app.py
├── model_utils.py
├── gradcam_utils.py
├── Pneumonia_resnet50.keras
├── Pneumonia_resnet50_metadata.json
├── requirements.txt
└── README.md




File Descriptions
app.py

This is the main Streamlit application file. It controls the user interface, image upload, model prediction, probability display, Grad-CAM generation, and downloadable result output.

Main responsibilities:

Render the web interface
Load the trained model and metadata
Accept uploaded chest X-ray images
Run prediction using the trained ResNet50 model
Display class probabilities and confidence score
Generate Grad-CAM explanations
Display original and Grad-CAM overlay images
model_utils.py

This file contains model loading, metadata validation, image preprocessing, and prediction functions.

Main responsibilities:

Load the trained .keras model
Load and validate metadata
Handle custom Lambda layer used during training
Convert uploaded images to RGB format
Resize images to the model input size
Run model inference
Return prediction class, confidence, pneumonia probability, and class probabilities

Expected class order:

["Normal", "Pneumonia-Bacterial", "Pneumonia-Viral"]
gradcam_utils.py

This file contains the Grad-CAM implementation used to explain model predictions.

Main responsibilities:

Build a Grad-CAM model from the trained CNN
Locate the target convolutional layer
Compute gradients for the selected class
Generate Grad-CAM heatmaps
Overlay the heatmap on the original chest X-ray image
Return visualization outputs for Streamlit display

For ResNet50, the default target convolution layer is:

conv5_block3_out
Pneumonia_resnet50.keras

This is the trained ResNet50 model used for three-class pneumonia classification.

The model predicts one of the following classes:

Normal
Pneumonia-Bacterial
Pneumonia-Viral
Pneumonia_resnet50_metadata.json

This file stores important model configuration information required during deployment.

Expected format:

{
  "IMAGE_SIZE": 224,
  "CHANNELS": 3,
  "class_names": ["Normal", "Pneumonia-Bacterial", "Pneumonia-Viral"]
}

The class order in this file must match the output order of the trained model.

Model Information

The deployed model is based on ResNet50, a deep convolutional neural network architecture known for its residual learning mechanism. ResNet50 uses skip connections to reduce the vanishing gradient problem and allows deeper networks to be trained more effectively.

In this project, ResNet50 was used for three-class chest X-ray classification. The model was trained to distinguish between normal chest X-ray images, bacterial pneumonia images, and viral pneumonia images.


Output

After prediction, the system displays:

Predicted class
Prediction confidence
Diagnosis group:
Normal
Pneumonia
Pneumonia subtype:
Bacterial Pneumonia
Viral Pneumonia
Not applicable for Normal class
Total pneumonia probability
Probability of each class
Grad-CAM visualization
Downloadable Grad-CAM result image
Grad-CAM Explainability

Grad-CAM, or Gradient-weighted Class Activation Mapping, is used to explain the model’s prediction by highlighting the important regions of the image that contributed most strongly to the selected class output.

In this application, Grad-CAM can be generated for:

The predicted class
Normal
Pneumonia-Bacterial
Pneumonia-Viral

The Grad-CAM output helps users visually inspect whether the model focuses on relevant chest regions rather than irrelevant background areas.

The system displays:

Original Chest X-ray | Grad-CAM Overlay

This supports the thesis objective of combining classification performance with explainable artificial intelligence.