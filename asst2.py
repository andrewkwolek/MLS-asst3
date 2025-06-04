from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import vggish_params
from vggish_input import waveform_to_examples, wavfile_to_examples
import sys
import wget
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
from pathlib import Path
from tqdm.notebook import tqdm
from google.colab import drive
import random
import hashlib
import time
from keras.models import load_model
from tensorflow import keras
import tensorflow as tf
import IPython.display as ipd
import librosa.display
import librosa
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
!pip install wget


# Suppress warnings
warnings.filterwarnings('ignore')

# Mount Google Drive
try:
    drive.mount('/content/drive')
    print("Google Drive mounted successfully.")
except:
    print("Error mounting Google Drive. Please run this cell again.")

# Install required packages if needed

# Download VGGish dependencies (if not already present)
!mkdir - p vggish
!wget - O vggish/vggish_input.py https: // raw.githubusercontent.com/tensorflow/models/master/research/audioset/vggish/vggish_input.py
!wget - O vggish/vggish_params.py https: // raw.githubusercontent.com/tensorflow/models/master/research/audioset/vggish/vggish_params.py
!wget - O vggish/mel_features.py https: // raw.githubusercontent.com/tensorflow/models/master/research/audioset/vggish/mel_features.py
!wget - O vggish/resampy.py https: // raw.githubusercontent.com/bmcfee/resampy/master/resampy/core.py

!pip install resampy

# Import VGGish modules
sys.path.append('./vggish')

# Configuration
# Your 5 activity classes from Assignment 1
ACTIVITIES = ['laugh', 'cough', 'clap', 'knock', 'alarm']
ENVIRONMENTS = ['small', 'large']
NUM_INSTANCES = 10
SAMPLE_RATE = 16000  # VGGish uses 16kHz

# Ubicoustics labels
UBICOUSTICS_LABELS = {
    'dog-bark': 0, 'drill': 1, 'hazard-alarm': 2, 'phone-ring': 3, 'speech': 4,
    'vacuum': 5, 'baby-cry': 6, 'chopping': 7, 'cough': 8, 'door': 9,
    'water-running': 10, 'knock': 11, 'microwave': 12, 'shaver': 13, 'toothbrush': 14,
    'blender': 15, 'dishwasher': 16, 'doorbell': 17, 'flush': 18, 'hair-dryer': 19,
    'laugh': 20, 'snore': 21, 'typing': 22, 'hammer': 23, 'car-horn': 24,
    'engine': 25, 'saw': 26, 'cat-meow': 27, 'alarm-clock': 28, 'cooking': 29
}

IDX_TO_LABEL = {v: k for k, v in UBICOUSTICS_LABELS.items()}

# Mapping from your 5 activities to Ubicoustics classes
# This maps your activities to the closest Ubicoustics classes
UBICOUSTICS_MAPPING = {
    'laugh': 'laugh',            # Direct match (20)
    'cough': 'cough',            # Direct match (8)
    # Similar to knock (11) - no exact match for clapping
    'clap': 'knock',
    'knock': 'knock',            # Direct match (11)
    'alarm': 'hazard-alarm'      # Similar to hazard-alarm (2)
}

# Human-readable labels for display
TO_HUMAN_LABELS = {
    'dog-bark': "Dog Barking", 'drill': "Drill In-Use", 'hazard-alarm': "Hazard Alarm",
    'phone-ring': "Phone Ringing", 'speech': "Person Talking", 'vacuum': "Vacuum In-Use",
    'baby-cry': "Baby Crying", 'chopping': "Chopping", 'cough': "Coughing",
    'door': "Door In-Use", 'water-running': "Water Running", 'knock': "Knocking",
    'microwave': "Microwave In-Use", 'shaver': "Shaver In-Use", 'toothbrush': "Toothbrushing",
    'blender': "Blender In-Use", 'dishwasher': "Dishwasher In-Use", 'doorbell': "Doorbel In-Use",
    'flush': "Toilet Flushing", 'hair-dryer': "Hair Dryer In-Use", 'laugh': "Laughing",
    'snore': "Snoring", 'typing': "Typing", 'hammer': "Hammering", 'car-horn': "Car Honking",
    'engine': "Vehicle Running", 'saw': "Saw In-Use", 'cat-meow': "Cat Meowing",
    'alarm-clock': "Alarm Clock", 'cooking': "Utensils and Cutlery"
}

# Path configuration
BASE_PATH = f"/content/drive/MyDrive"
DATASET_PATH = f"{BASE_PATH}/12481"  # Replace with your dataset ID
MODEL_PATH = f"{BASE_PATH}/models"
CLASS_DATASET_PATH = f"{BASE_PATH}/class_dataset"

# Create necessary directories
os.makedirs(MODEL_PATH, exist_ok=True)

# Load or create your dataset ID


def get_dataset_id():
    # Replace this with your dataset ID from Assignment 1
    # Or uncomment the code below to generate a new one
    return "12481"

    student_name = input("Enter your full name: ")
    hash_object = hashlib.md5(student_name.encode())
    hex_dig = hash_object.hexdigest()
    seed = int(hex_dig, 16) % (10**8)
    random.seed(seed)
    return random.randint(10000, 99999)


dataset_id = get_dataset_id()
DATASET_PATH = f"{BASE_PATH}/{dataset_id}"
print(f"Using dataset path: {DATASET_PATH}")

# Mount Google Drive (directory of your audio data)
drive.mount('/content/drive')


# TODO: Load the pre-trained Ubicoustics model
# For example, if a model file is provided, you might use:
# model = load_model('ubicoustics_model.h5')
ubicoustics_model_path = f"{MODEL_PATH}/ubicoustics_model.h5"
model = load_model(ubicoustics_model_path, compile=False)
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy')


# TODO: Mapping from Ubicoustics classes to your classes
REVERSE_UBICOUSTICS_MAPPING = {
    'laugh': 'laugh',            # Direct match (20)
    'cough': 'cough',            # Direct match (8)
    'knock': 'knock',            # Direct match (11)
    'hazard-alarm': 'alarm'      # Similar to hazard-alarm (2)

}

# TODO: Prepare your data: list all audio file paths and their true labels
audio_files = []    # List of file paths for your 100 audio recordings
# List of true labels (e.g., 'cough', 'laugh', etc.) for each recording
true_labels = []
# List of environment labels ('small' or 'large') for each recording (parallel to the files)
environments = []

if os.path.exists(DATASET_PATH):
    for root, dirs, files in os.walk(DATASET_PATH):
        for file in files:
            if file.endswith('.wav'):
                # Check if the file follows the naming convention
                file_parts = os.path.splitext(file)[0].split('_')

                # Handle different naming conventions
                if len(file_parts) >= 2:
                    activity = file_parts[0]
                    environment = file_parts[1]

                    # Check if activity and environment are valid
                    if activity in ACTIVITIES and environment in ENVIRONMENTS:
                        audio_files.append(os.path.join(root, file))
                        true_labels.append(activity)
                        environments.append(environment)
                    else:
                        print(
                            f"Skipping {file}: invalid activity or environment")
                else:
                    print(f"Skipping {file}: invalid naming format")

print(f"Found {len(audio_files)} audio files for processing")


def predict_ubicoustics(file_path):
    # Convert audio to VGGish input format
    try:
        # Use librosa to load the audio file instead of direct wavfile_to_examples
        # This is more robust with different audio formats
        y, sr = librosa.load(
            file_path, sr=vggish_params.SAMPLE_RATE, mono=True)

        # Convert the loaded audio to VGGish examples format
        examples = waveform_to_examples(y, sr)

        if examples.shape[0] == 0:
            print(f"Warning: No examples generated from {file_path}")
            return None

        # Run prediction with Ubicoustics model
        predictions = model.predict(examples)

        # Average predictions across all windows
        avg_pred = np.mean(predictions, axis=0)

        # Get top class
        top_class_idx = np.argmax(avg_pred)

        # Convert to Ubicoustics label
        ubicoustics_labels_reversed = {
            v: k for k, v in UBICOUSTICS_LABELS.items()}
        predicted_ubicoustics_class = ubicoustics_labels_reversed[top_class_idx]

        # If the predicted class is in your mapping, return it
        # Otherwise, find the closest match
        if predicted_ubicoustics_class in REVERSE_UBICOUSTICS_MAPPING:
            # If there are multiple of your classes mapped to this Ubicoustics class,
            # just choose the first one (this is a simplification)
            return REVERSE_UBICOUSTICS_MAPPING[predicted_ubicoustics_class]
        else:
            # If no direct mapping, return None or some default
            print(
                f"Warning: No mapping for Ubicoustics class: {predicted_ubicoustics_class}")
            return predicted_ubicoustics_class

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None


# TODO: Run the model on each audio file and collect predictions
pred_labels = []
print("Running predictions on all audio files...")

for file_path in tqdm(audio_files):
    prediction = predict_ubicoustics(file_path)
    pred_labels.append(prediction)

# TODO: Calculate and report the overall accuracy
accuracy = accuracy_score(true_labels, pred_labels)
print(f"Overall accuracy: {accuracy:.4f}")

# TODO: Compute a confusion matrix for the predictions vs. true labels
cm = confusion_matrix(true_labels, pred_labels, labels=ACTIVITIES)
print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=ACTIVITIES, yticklabels=ACTIVITIES)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Pre-trained Ubicoustics Model')
plt.show()

# report = classification_report(true_labels, pred_labels, labels=ACTIVITIES)
# print("Classification Report:")
# print(report)

# TODO: Compare performance across your two environments:
small_indices = [i for i, env in enumerate(environments) if env == 'small']
large_indices = [i for i, env in enumerate(environments) if env == 'large']

small_true = [true_labels[i] for i in small_indices]
small_pred = [pred_labels[i] for i in small_indices]
large_true = [true_labels[i] for i in large_indices]
large_pred = [pred_labels[i] for i in large_indices]

small_accuracy = accuracy_score(small_true, small_pred)
large_accuracy = accuracy_score(large_true, large_pred)

# Create confusion matrices for each environment
small_cm = confusion_matrix(small_true, small_pred, labels=ACTIVITIES)
large_cm = confusion_matrix(large_true, large_pred, labels=ACTIVITIES)

# Plot confusion matrices for each environment
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

sns.heatmap(small_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=ACTIVITIES, yticklabels=ACTIVITIES, ax=axes[0])
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')
axes[0].set_title('Small Room Confusion Matrix')

sns.heatmap(large_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=ACTIVITIES, yticklabels=ACTIVITIES, ax=axes[1])
axes[1].set_xlabel('Predicted Label')
axes[1].set_ylabel('True Label')
axes[1].set_title('Large Room Confusion Matrix')

plt.tight_layout()
plt.show()

small_indices = [i for i, env in enumerate(environments) if env == 'small']
large_indices = [i for i, env in enumerate(environments) if env == 'large']

small_files = [audio_files[i] for i in small_indices]
small_labels = [true_labels[i] for i in small_indices]
large_files = [audio_files[i] for i in large_indices]
large_labels = [true_labels[i] for i in large_indices]

print(f"Small room dataset: {len(small_files)} files")
print(f"Large room dataset: {len(large_files)} files")

# Define a function to prepare VGGish examples from audio files


def prepare_examples_from_files(file_paths, labels):
    X = []  # Will hold all our audio examples
    y = []  # Will hold integer labels corresponding to activities

    # Map activity names to integer indices
    activity_to_idx = {activity: idx for idx,
                       activity in enumerate(ACTIVITIES)}

    for i, file_path in enumerate(tqdm(file_paths, desc="Processing audio files")):
        try:
            # Load audio
            audio, sr = librosa.load(
                file_path, sr=vggish_params.SAMPLE_RATE, mono=True)

            # Convert to VGGish format
            examples = waveform_to_examples(audio, sr)

            if examples.shape[0] > 0:  # If we got valid examples
                # Store examples
                X.append(examples)

                # Store the same class index for all examples from this file
                class_idx = activity_to_idx[labels[i]]
                y.append(np.full(examples.shape[0], class_idx))
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    # Combine all examples into one array
    if X:
        X = np.vstack(X)
        y = np.concatenate(y)
        return X, y
    else:
        print("No valid examples were generated!")
        return None, None


# Process data for each environment
X_small, y_small = prepare_examples_from_files(small_files, small_labels)
X_large, y_large = prepare_examples_from_files(large_files, large_labels)

print(f"Small room data shape: {X_small.shape}, Labels shape: {y_small.shape}")
print(f"Large room data shape: {X_large.shape}, Labels shape: {y_large.shape}")

# TODO: Fine-tune model and experiment with different hyperparameters


def create_fine_tuning_model(base_model, num_layers_to_freeze=0):
    """
    Create a model for fine-tuning based on base model.

    Args:
        base_model: Base Ubicoustics model
        num_layers_to_freeze: Number of layers to freeze from the beginning

    Returns:
        Model for fine-tuning
    """
    # Clone the base model
    model = keras.models.clone_model(base_model)
    model.set_weights(base_model.get_weights())

    # Total number of layers
    total_layers = len(model.layers)

    # Freeze layers if specified
    if num_layers_to_freeze > 0:
        # Special case for last layer only
        if num_layers_to_freeze == -1:
            # Freeze all layers except the last one
            for layer in model.layers[:-1]:
                layer.trainable = False
            model.layers[-1].trainable = True
            print(f"Freezing all layers except the last one")
        else:
            # Freeze specified number of layers from the beginning
            for layer in model.layers[:num_layers_to_freeze]:
                layer.trainable = False
            print(
                f"Freezing {num_layers_to_freeze} layers out of {total_layers}")
    else:
        print("All layers are trainable")

    # Count trainable parameters
    trainable_count = sum(K.count_params(w) for w in model.trainable_weights)
    non_trainable_count = sum(K.count_params(w)
                              for w in model.non_trainable_weights)

    print(f"Total parameters: {trainable_count + non_trainable_count}")
    print(f"Trainable parameters: {trainable_count}")
    print(f"Non-trainable parameters: {non_trainable_count}")

    # Compile model
    model.compile(
        optimizer='adam',  # Will be replaced during fine-tuning
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def fine_tune_model(model, X_train, y_train, X_test, y_test, learning_rate, num_epochs):
    """
    Fine-tune model on given data.

    Args:
        model: Model to fine-tune
        X_train, y_train: Training data
        X_test, y_test: Testing data
        learning_rate: Learning rate for optimization
        num_epochs: Number of training epochs

    Returns:
        Fine-tuned model and training history
    """
    # Create optimizer with specified learning rate
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Create callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]

    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=num_epochs,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    return model, history

# Function to get file-level predictions from a model


def predict_on_files(model, file_paths, true_labels):
    """
    Get file-level predictions by aggregating window-level predictions

    Args:
        model: The trained model
        file_paths: List of audio file paths to predict on
        true_labels: List of true labels for each file

    Returns:
        List of predicted labels, list of true labels
    """
    # For storing results
    true_file_labels = []
    pred_file_labels = []

    # Map activity names to indices and back
    activity_to_idx = {activity: idx for idx,
                       activity in enumerate(ACTIVITIES)}
    idx_to_activity = {idx: activity for idx,
                       activity in enumerate(ACTIVITIES)}

    # Process each file
    for i, file_path in enumerate(tqdm(file_paths, desc="Predicting on files")):
        try:
            # Get true label
            true_label = true_labels[i]
            true_file_labels.append(true_label)

            # Load and process audio
            audio, sr = librosa.load(
                file_path, sr=vggish_params.SAMPLE_RATE, mono=True)
            examples = waveform_to_examples(audio, sr)

            if examples.shape[0] > 0:
                # Get predictions for all windows
                window_preds = model.predict(examples)

                # Average predictions across windows
                avg_pred = np.mean(window_preds, axis=0)

                # Get the class with highest average prediction
                pred_idx = np.argmax(avg_pred)
                pred_label = idx_to_activity[pred_idx]

                # Store prediction
                pred_file_labels.append(pred_label)
            else:
                print(f"Warning: No valid examples for {file_path}")
                # If no valid examples, skip this file
                true_file_labels.pop()  # Remove the corresponding true label
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            # If error, skip this file
            true_file_labels.pop()  # Remove the corresponding true label

    return pred_file_labels, true_file_labels

# Modified fine-tuning function that evaluates at file level


def fine_tune_and_evaluate(model, train_files, train_labels, test_files, test_labels,
                           learning_rate, epochs, freeze_all_but_last=False):
    """
    Fine-tune model and evaluate at the file level

    Args:
        model: Base model to fine-tune
        train_files, train_labels: Training data files and labels
        test_files, test_labels: Test data files and labels
        learning_rate, epochs, freeze_all_but_last: Training parameters

    Returns:
        Trained model, test accuracy, and confusion matrix
    """
    X_train, y_train = prepare_examples_from_files(train_files, train_labels)

    clone = keras.models.clone_model(model)
    clone.set_weights(model.get_weights())

    if freeze_all_but_last:
        for layer in clone.layers[:-1]:
            layer.trainable = False
        clone.layers[-1].trainable = True
        print("Freezing all layers except the last one")
    else:
        print("All layers are trainable")

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    clone.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Use a small portion of training data for validation
    history = clone.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.1,  # Use 10% of training data for validation
        verbose=1
    )

    pred_labels, true_labels = predict_on_files(clone, test_files, test_labels)

    accuracy = accuracy_score(true_labels, pred_labels)
    cm = confusion_matrix(true_labels, pred_labels, labels=ACTIVITIES)

    print(f"File-level test accuracy: {accuracy:.4f}")

    return clone, accuracy, cm, history


# Run experiments with corrected evaluation
# Initialize results tracking
results = []

# Define the runs you want to try manually
manual_runs = [
    {
        "direction": "large → small",
        "train_files": large_files,
        "train_labels": large_labels,
        "test_files": small_files,
        "test_labels": small_labels,
        "learning_rate": 0.001,
        "epochs": 10,
        "freeze_all_but_last": True,
        "strategy_name": "Last layer only"
    },
]


# Run each manual configuration
for run in manual_runs:
    print(f"\n{run['direction']} | Strategy: {run['strategy_name']} | LR: {run['learning_rate']} | Epochs: {run['epochs']}")

    # Fine-tune and evaluate
    trained_model, accuracy, cm, history = fine_tune_and_evaluate(
        model,
        run['train_files'], run['train_labels'],
        run['test_files'], run['test_labels'],
        run['learning_rate'], run['epochs'],
        run['freeze_all_but_last']
    )

    # Store result
    results.append({
        'direction': run['direction'],
        'strategy': run['strategy_name'],
        'learning_rate': run['learning_rate'],
        'epochs': run['epochs'],
        'accuracy': accuracy
    })

    print(f"{run['direction']}: {run['strategy_name']}, LR={run['learning_rate']}, Epochs={run['epochs']}\nAccuracy: {accuracy:.4f}")


def load_holdout_data(holdout_data_path):
    """
    Load the holdout dataset, returning file paths and labels with a progress bar.

    Args:
        holdout_data_path (str): Path to the root directory of the holdout dataset.

    Returns:
        tuple: A tuple containing two lists:
            - file_paths (list): List of audio file paths.
            - labels (list): List of labels corresponding to each file.
    """
    file_paths = []
    labels = []

    # Walk through each subdirectory (activity)
    for activity in tqdm(os.listdir(holdout_data_path), desc="Loading data", unit="activity"):
        activity_path = os.path.join(holdout_data_path, activity)

        # Check if it's a directory (it should be, since each directory is labeled with an activity)
        if os.path.isdir(activity_path):
            # Iterate over all files in the subdirectory
            for filename in tqdm(os.listdir(activity_path), desc=f"Processing {activity}", unit="file", leave=False):
                if filename.endswith(".wav"):  # Assuming audio files are .wav format
                    file_path = os.path.join(activity_path, filename)
                    file_paths.append(file_path)
                    # The subdirectory name is the label
                    labels.append(activity)

    return file_paths, labels


# TODO: Load the holdout dataset (from A1.6 Dropbox link)
holdout_files = []  # List to store file paths
holdout_labels = []  # List to store true labels

holdout_data_path = '/content/drive/MyDrive/holdout_set'
holdout_files, holdout_labels = load_holdout_data(holdout_data_path)

# TODO: Train on all your collected data using best hyperparameters from A2.2
trained_model, accuracy, cm, history = fine_tune_and_evaluate(
    model,
    train_files=large_files,
    train_labels=large_labels,
    test_files=holdout_files,
    test_labels=holdout_labels,
    learning_rate=0.001,
    epochs=10,
    freeze_all_but_last=True
)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=ACTIVITIES, yticklabels=ACTIVITIES)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f"{run['direction']}: {run['strategy_name']}, LR={run['learning_rate']}, Epochs={run['epochs']}\nAccuracy: {accuracy:.4f}")
plt.tight_layout()
plt.show()

print(holdout_labels)

ubicoustics_model_path = f"{MODEL_PATH}/ubicoustics_model.h5"
base_model = load_model(ubicoustics_model_path, compile=False)
base_model.compile(optimizer=Adam(learning_rate=1e-4),
                   loss='categorical_crossentropy')

pred_labels = []
print("Running predictions on all audio files...")

for file_path in tqdm(holdout_files):
    prediction = predict_ubicoustics(file_path)
    pred_labels.append(prediction)

accuracy = accuracy_score(holdout_labels, pred_labels)
print(accuracy)
cm = confusion_matrix(holdout_labels, pred_labels, labels=ACTIVITIES)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=ACTIVITIES, yticklabels=ACTIVITIES)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f"Base Model")
plt.tight_layout()
plt.show()
