import nbformat as nbf

# Initialize a new notebook object
nb = nbf.v4.new_notebook()

cells = []

# Cell 1 (Markdown)
cells.append(nbf.v4.new_markdown_cell("""# DeepFake Detection using Convolutional Neural Networks (CNN)

## 📌 Project Overview

DeepFake technology uses Artificial Intelligence to manipulate videos by replacing faces or altering expressions, making it difficult to distinguish fake content from genuine videos.

This project aims to build a DeepFake Detection System using a Convolutional Neural Network (CNN). The model is trained on the Celeb-DF dataset by extracting frames from real and fake videos. Each frame is resized, normalized, and used to train the CNN model to classify videos as either **Real** or **Fake**.

---

## 🎯 Objectives

- Extract frames from videos
- Calculate frame editing percentage
- Train a CNN model
- Detect DeepFake videos
- Evaluate model performance
- Save the trained model

---

## 🗂 Dataset

Dataset Used: **Celeb-DF**

Folder Structure

Celeb-DF/
│
├── Celeb-real/
│ ├── video1.mp4
│ ├── video2.mp4
│
├── Celeb-synthesis/
│ ├── fake1.mp4
│ ├── fake2.mp4

---

## Libraries Used

- OpenCV
- NumPy
- TensorFlow
- Scikit-Learn
- Matplotlib"""))

# Cell 2 (Code)
cells.append(nbf.v4.new_code_cell("""# ==========================
# Import Required Libraries
# ==========================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

print("Libraries Imported Successfully")"""))

# Cell 3 (Code)
cells.append(nbf.v4.new_code_cell("""# ==========================
# Dataset Paths
# ==========================

REAL_VIDEOS_PATH = r'D:\\Celeb-DF\\Celeb-real'
FAKE_VIDEOS_PATH = r'D:\\Celeb-DF\\Celeb-synthesis'

print("Real Dataset :", REAL_VIDEOS_PATH)
print("Fake Dataset :", FAKE_VIDEOS_PATH)"""))

# Cell 4 (Code)
cells.append(nbf.v4.new_code_cell("""# ==========================
# Hyperparameters
# ==========================

IMG_SIZE = 128

NUM_FRAMES = 10

BATCH_SIZE = 16

EPOCHS = 10

TEST_SIZE = 0.20

print("Image Size :", IMG_SIZE)
print("Frames per Video :", NUM_FRAMES)
print("Batch Size :", BATCH_SIZE)
print("Epochs :", EPOCHS)"""))

# Cell 5 (Code)
cells.append(nbf.v4.new_code_cell("""# ==========================
# Verify Dataset
# ==========================

# Use try-except to handle cases where paths do not exist during dry run
try:
    print("Number of Real Videos :", len(os.listdir(REAL_VIDEOS_PATH)))
    print("Number of Fake Videos :", len(os.listdir(FAKE_VIDEOS_PATH)))
except FileNotFoundError:
    print("Number of Real Videos : 0 (Directory not found - running in dry environment)")
    print("Number of Fake Videos : 0 (Directory not found - running in dry environment)")"""))

# Cell 6 (Markdown)
cells.append(nbf.v4.new_markdown_cell("""# Frame Extraction

## Why Frame Extraction?

A video is simply a sequence of images called frames.

Instead of training directly on videos, we extract multiple representative frames from each video.

Advantages:

- Faster training
- Less memory consumption
- Better feature learning
- Easier preprocessing

Each extracted frame is resized to **128 × 128 pixels** before being fed into the CNN."""))

# Cell 7 (Code)
cells.append(nbf.v4.new_code_cell("""# ============================================
# Function to Extract Frames from Videos
# ============================================

def extract_frames(video_path, num_frames=10):

    frames = []

    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        print(f"Unable to read video: {video_path}")
        cap.release()
        return np.array(frames), 0

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        num_frames,
        dtype=int
    )

    prev_frame = None

    frame_differences = []

    for idx in frame_indices:

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

        ret, frame = cap.read()

        if ret:

            frame = cv2.resize(
                frame,
                (IMG_SIZE, IMG_SIZE)
            )

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            frames.append(frame)

            if prev_frame is not None:

                diff = cv2.absdiff(
                    prev_frame,
                    gray
                )

                frame_differences.append(
                    np.mean(diff)
                )

            prev_frame = gray

    cap.release()

    if frame_differences:

        editing_percentage = (
            np.mean(frame_differences) / 255
        ) * 100

    else:

        editing_percentage = 0

    return np.array(frames), editing_percentage"""))

# Cell 8 (Markdown)
cells.append(nbf.v4.new_markdown_cell("""# Loading the Dataset

## Overview

After extracting frames from the videos, we need to organize them into a dataset.

Each extracted frame is assigned a label:

- **0 → Real Video**
- **1 → Fake Video**

The dataset consists of:

- X → Image Frames
- y → Labels

We also calculate the average editing percentage for each video."""))

# Cell 9 (Code)
cells.append(nbf.v4.new_code_cell("""# ==========================================
# Load Dataset Function
# ==========================================

def load_data(real_videos_path, fake_videos_path, num_frames=10):

    X = []
    y = []
    editing_stats = []

    # --------------------------
    # Load Real Videos
    # --------------------------

    print("Loading Real Videos...\\n")

    try:
        real_list = os.listdir(real_videos_path)
    except FileNotFoundError:
        real_list = []

    for video in real_list:

        video_path = os.path.join(real_videos_path, video)

        if not os.path.isfile(video_path):
            continue

        frames, edit_percent = extract_frames(
            video_path,
            num_frames
        )

        for frame in frames:
            X.append(frame)
            y.append(0)

        editing_stats.append(edit_percent)

    # --------------------------
    # Load Fake Videos
    # --------------------------

    print("Loading Fake Videos...\\n")

    try:
        fake_list = os.listdir(fake_videos_path)
    except FileNotFoundError:
        fake_list = []

    for video in fake_list:

        video_path = os.path.join(fake_videos_path, video)

        if not os.path.isfile(video_path):
            continue

        frames, edit_percent = extract_frames(
            video_path,
            num_frames
        )

        for frame in frames:
            X.append(frame)
            y.append(1)

        editing_stats.append(edit_percent)

    print("------------------------------------")
    print("Dataset Loading Finished")
    print("------------------------------------")

    avg_edit = round(np.mean(editing_stats), 2) if editing_stats else 0.0
    print("Average Editing Percentage :", avg_edit, "%")

    return np.array(X), np.array(y)"""))

# Cell 10 (Code)
cells.append(nbf.v4.new_code_cell("""# ==========================================
# Load Dataset
# ==========================================

X, y = load_data(
    REAL_VIDEOS_PATH,
    FAKE_VIDEOS_PATH,
    NUM_FRAMES
)

# Mock some data if running in an empty environment for pipeline demonstration
if len(X) == 0:
    print("\\n[NOTE] No files found. Generating mock dataset for execution demonstration purposes...")
    X = np.random.randint(0, 256, size=(100, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    y = np.random.choice([0, 1], size=(100,))

print("\\nDataset Summary:")
print("Images Shape :", X.shape)
print("Labels Shape :", y.shape)"""))

# Cell 11 (Markdown)
cells.append(nbf.v4.new_markdown_cell("""# Data Preprocessing

Before training the CNN, the images must be preprocessed.

### Steps

- Normalize pixel values
- Convert labels into float values
- Prepare images for CNN input

Normalization scales pixel values from **0–255** to **0–1**, which helps the neural network converge faster."""))

# Cell 12 (Code)
cells.append(nbf.v4.new_code_cell("""# ==========================================
# Data Preprocessing
# ==========================================

def preprocess_data(X, y):

    X = X.astype("float32")

    X = X / 255.0

    y = y.astype("float32")

    return X, y"""))

# Cell 13 (Code)
cells.append(nbf.v4.new_code_cell("""# ==========================================
# Run Preprocessing
# ==========================================

X, y = preprocess_data(X, y)
print("Normalization Completed")
print("Maximum Pixel Value :", np.max(X))
print("Minimum Pixel Value :", np.min(X))"""))

# Cell 14 (Markdown)
cells.append(nbf.v4.new_markdown_cell("""# Train-Test Split

The dataset is divided into:

- **80% Training Data**
- **20% Testing Data**

Training data is used to learn patterns.

Testing data is used to evaluate the model on unseen data."""))

# Cell 15 (Code)
cells.append(nbf.v4.new_code_cell("""# ==========================================
# Split Dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=42,
    shuffle=True
)

print("Training Images :", X_train.shape)
print("Testing Images :", X_test.shape)
print("Training Labels :", y_train.shape)
print("Testing Labels :", y_test.shape)"""))

# Cell 16 (Code)
cells.append(nbf.v4.new_code_cell("""# ==========================================
# Display Sample Frames
# ==========================================

plt.figure(figsize=(12,6))
for i in range(min(6, len(X_train))):

    plt.subplot(2,3,i+1)

    plt.imshow(X_train[i])

    plt.axis("off")
plt.suptitle("Sample Training Frames")
plt.show()"""))

# Cell 17 (Markdown)
cells.append(nbf.v4.new_markdown_cell("""# Dataset Summary

After preprocessing:

- Images are normalized.
- Labels are prepared.
- Dataset is split into training and testing sets.

The data is now ready to be fed into the CNN model."""))

# Cell 18 (Markdown)
cells.append(nbf.v4.new_markdown_cell("""# Building the CNN Model

## What is a CNN?

A Convolutional Neural Network (CNN) is a deep learning model designed to process image data. It automatically learns important visual features such as edges, textures, shapes, and patterns.

### CNN Architecture

Input Image (128×128×3)
↓
Conv2D (64 Filters)
↓
Batch Normalization
↓
Max Pooling
↓
Conv2D (128 Filters)
↓
Batch Normalization
↓
Max Pooling
↓
Conv2D (256 Filters)
↓
Batch Normalization
↓
Max Pooling
↓
Flatten
↓
Dense (128)
↓
Dropout (0.5)
↓
Output (Sigmoid)

The output layer predicts:

- **0 → Real Video**
- **1 → Fake Video**"""))

# Cell 19 (Code)
cells.append(nbf.v4.new_code_cell("""# ==========================================
# Build CNN Model
# ==========================================

def build_model(input_shape):

    model = models.Sequential([

        layers.Input(shape=input_shape),

        layers.Conv2D(
            64,
            (3,3),
            activation='relu'
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D((2,2)),

        layers.Conv2D(
            128,
            (3,3),
            activation='relu'
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D((2,2)),

        layers.Conv2D(
            256,
            (3,3),
            activation='relu'
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D((2,2)),

        layers.Flatten(),

        layers.Dense(
            128,
            activation='relu'
        ),

        layers.BatchNormalization(),

        layers.Dropout(0.5),

        layers.Dense(
            1,
            activation='sigmoid'
        )

    ])

    model.compile(

        optimizer='adam',

        loss='binary_crossentropy',

        metrics=['accuracy']

    )

    return model"""))

# Cell 20 (Code)
cells.append(nbf.v4.new_code_cell("""# ==========================================
# Create Model
# ==========================================

model = build_model(
    (IMG_SIZE, IMG_SIZE, 3)
)
print("CNN Model Created Successfully")"""))

# Cell 21 (Code)
cells.append(nbf.v4.new_code_cell("""# ==========================================
# Model Summary
# ==========================================

model.summary()"""))

# Cell 22 (Markdown)
cells.append(nbf.v4.new_markdown_cell("""# Model Training

The CNN model is trained using the training dataset.

### Optimizer

Adam Optimizer

### Loss Function

Binary Cross Entropy

### Metric

Accuracy

### Early Stopping

Early Stopping prevents overfitting by stopping training when the validation loss stops improving."""))

# Cell 23 (Code)
cells.append(nbf.v4.new_code_cell("""# ==========================================
# Train Model
# ==========================================

def train_model(
        model,
        X_train,
        y_train,
        epochs,
        batch_size
):

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.20,
        callbacks=[early_stop],
        verbose=1
    )

    return history"""))

# Cell 24 (Code)
cells.append(nbf.v4.new_code_cell("""# ==========================================
# Start Training
# ==========================================

print("Training Started...\\n")
history = train_model(
    model,
    X_train,
    y_train,
    EPOCHS,
    BATCH_SIZE
)
print("\\nTraining Completed Successfully")"""))

# Cell 25 (Markdown)
cells.append(nbf.v4.new_markdown_cell("""# Training History

The training history stores:

- Training Accuracy
- Validation Accuracy
- Training Loss
- Validation Loss

These values will be used to visualize the model's learning performance."""))

# Cell 26 (Code)
cells.append(nbf.v4.new_code_cell("""# ==========================================
# Display Training Metrics
# ==========================================

print("Training Accuracy :")
print(history.history['accuracy'])
print("\\nValidation Accuracy :")
print(history.history['val_accuracy'])
print("\\nTraining Loss :")
print(history.history['loss'])
print("\\nValidation Loss :")
print(history.history['val_loss'])"""))

# Cell 27 (Markdown)
cells.append(nbf.v4.new_markdown_cell("""# Model Evaluation

After training, we evaluate the model using the unseen test dataset.

The evaluation metrics include:

- Test Accuracy
- Test Loss
- Confusion Matrix
- Classification Report

These metrics help determine how well the model generalizes to new data."""))

# Cell 28 (Code)
cells.append(nbf.v4.new_code_cell("""# ==========================================
# Evaluate Model
# ==========================================

loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print("\\n========== Model Evaluation ==========")
print(f"Test Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy*100:.2f}%")"""))

# Cell 29 (Markdown)
cells.append(nbf.v4.new_markdown_cell("""# Training Accuracy Graph

The following graph shows how the training and validation accuracy change over each epoch."""))

# Cell 30 (Code)
cells.append(nbf.v4.new_code_cell("""# ==========================================
# Plot Accuracy
# ==========================================

plt.figure(figsize=(8,5))
plt.plot(
    history.history['accuracy'],
    label='Training Accuracy',
    linewidth=2
)
plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy',
    linewidth=2
)
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()"""))

# Cell 31 (Markdown)
cells.append(nbf.v4.new_markdown_cell("""# Training Loss Graph

This graph shows how the training and validation loss decrease during training."""))

# Cell 32 (Code)
cells.append(nbf.v4.new_code_cell("""# ==========================================
# Plot Loss
# ==========================================

plt.figure(figsize=(8,5))
plt.plot(
    history.history['loss'],
    label='Training Loss',
    linewidth=2
)
plt.plot(
    history.history['val_loss'],
    label='Validation Loss',
    linewidth=2
)
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()"""))

# Cell 33 (Code)
cells.append(nbf.v4.new_code_cell("""# ==========================================
# Predict Test Dataset
# ==========================================

predictions = model.predict(X_test)
predictions = (predictions > 0.5).astype(int)"""))

# Cell 34 (Code)
cells.append(nbf.v4.new_code_cell("""# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Real","Fake"]
)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()"""))

# Cell 35 (Code)
cells.append(nbf.v4.new_code_cell("""# ==========================================
# Classification Report
# ==========================================

print(classification_report(
    y_test,
    predictions,
    target_names=["Real","Fake"]
))"""))

# Cell 36 (Markdown)
cells.append(nbf.v4.new_markdown_cell("""# Save Trained Model

The trained CNN model is saved for future prediction without retraining."""))

# Cell 37 (Code)
cells.append(nbf.v4.new_code_cell("""# ==========================================
# Save Model
# ==========================================

model.save("deepfake_detection_model.h5")
print("Model Saved Successfully")"""))

# Cell 38 (Code)
cells.append(nbf.v4.new_code_cell("""# ==========================================
# Predict a Single Video
# ==========================================

def predict_video(video_path):

    frames, _ = extract_frames(video_path)

    if len(frames) == 0:

        print("Unable to load video.")

        return

    frames = frames.astype("float32") / 255.0

    predictions = model.predict(frames)

    average_prediction = np.mean(predictions)

    print("\\nAverage Prediction Score :", average_prediction)

    if average_prediction >= 0.5:

        print("Prediction : FAKE VIDEO")

    else:

        print("Prediction : REAL VIDEO")"""))

# Cell 39 (Code)
cells.append(nbf.v4.new_code_cell("""# ==========================================
# Example Prediction
# ==========================================

video_path = r"D:\\Celeb-DF\\Celeb-synthesis\\sample.mp4"
# Uncomment to test
# predict_video(video_path)"""))

# Cell 40 (Markdown)
cells.append(nbf.v4.new_markdown_cell("""# Project Conclusion

## Summary

In this project, a Convolutional Neural Network (CNN) was developed to detect DeepFake videos using the Celeb-DF dataset.

### Workflow

- Video Loading
- Frame Extraction
- Image Preprocessing
- CNN Training
- Model Evaluation
- DeepFake Prediction

### Result

The model successfully learned visual patterns from extracted frames and classified videos into **Real** and **Fake** categories.

### Future Improvements

- Transfer Learning (ResNet50 / EfficientNet)
- Face Detection before classification
- Attention Mechanism
- Temporal Feature Extraction using LSTM
- Vision Transformer (ViT)
- Real-time webcam detection"""))

# Pack all the cells into the notebook object
nb['cells'] = cells

# Define file path and save
filename = 'DeepFake_Detection_CNN.ipynb'
with open(filename, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Notebook fully generated and saved as: {filename}")