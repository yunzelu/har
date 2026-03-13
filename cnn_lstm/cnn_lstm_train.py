import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tqdm import tqdm
# Add these imports for validation metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from itertools import cycle

# Mount Google Drive
from google.colab import drive

drive.mount('/content/drive')

# Configuration
dataset_path = '/content/drive/MyDrive/T01/Keypoints2'
activities = ['Clapping', 'Sitting', 'Standing Still',
              'Walking While Reading Book', 'Walking While Using Phone', 'Walking']
sequence_length = 30

# Store data and labels
data = []
labels = []


# Function to load frames from a single JSON file
def load_normalized_json_sequence(json_path, activity, sequence_length=30):
    with open(json_path, 'r') as f:
        frames = json.load(f)

    keypoints_sequence = []
    data_out = []
    labels_out = []

    for frame in frames:
        if frame and 'keypoints' in frame[0]:
            keypoints = frame[0]['keypoints']
            keypoints_flat = [val for point in keypoints for val in point]
        else:
            keypoints_flat = [0] * 34  # 17 keypoints × 2

        keypoints_sequence.append(keypoints_flat)

        if len(keypoints_sequence) == sequence_length:
            data_out.append(np.array(keypoints_sequence).flatten())
            labels_out.append(activity)
            keypoints_sequence = []

    return data_out, labels_out


# Read each activity and its files
for activity in activities:
    activity_folder = os.path.join(dataset_path, activity)
    if not os.path.isdir(activity_folder):
        print(f"Missing activity folder: {activity}")
        continue

    files = [f for f in os.listdir(activity_folder) if f.endswith('.json')]

    for file in tqdm(files, desc=f"Processing {activity}"):
        file_path = os.path.join(activity_folder, file)
        d, l = load_normalized_json_sequence(file_path, activity, sequence_length)
        data.extend(d)
        labels.extend(l)

# Debug output
print(f"Total sequences collected: {len(data)}")
print(f"Sample labels: {labels[:5]}")

# DataFrame conversion
data_df = pd.DataFrame(data)
data_df['activity'] = labels

# Separate features and labels
X = data_df.drop('activity', axis=1).values
y = data_df['activity'].values

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
# Store original class names for later use in visualization
class_names = label_encoder.classes_
y_categorical = to_categorical(y_encoded)

# Save the label encoder mapping to a text file
encoder_mapping = {idx: label for idx, label in enumerate(label_encoder.classes_)}
encoder_file_path = '/content/drive/MyDrive/T01/activity_classes.txt'

# Save as a JSON file for easy reading and loading
with open(encoder_file_path, 'w') as f:
    json.dump(encoder_mapping, f, indent=4)

print(f"Label encoder mapping saved to: {encoder_file_path}")


# Define function to load the label encoder mapping (for future use)
def load_label_encoder_mapping(file_path):
    with open(file_path, 'r') as f:
        mapping = json.load(f)

    # Convert keys back to integers (JSON serializes them as strings)
    mapping = {int(k): v for k, v in mapping.items()}

    # Create a new label encoder
    le = LabelEncoder()
    # Set the classes_ attribute to match the original
    le.classes_ = np.array(list(mapping.values()))

    return le


# Train-test split - IMPORTANT: we need to keep original labels for later evaluation
X_train, X_test, y_train_cat, y_test_cat, y_train_enc, y_test_enc = train_test_split(
    X, y_categorical, y_encoded, test_size=0.2, random_state=42)

# Reshape for CNN-LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build model
model = Sequential()
model.add(Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(MaxPooling1D(2))
model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Conv1D(256, 3, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Dropout(0.3))

model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(y_categorical.shape[1], activation='softmax'))

# Model summary
model.summary()

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train_cat, epochs=50, batch_size=32, validation_data=(X_test, y_test_cat))

# Evaluate basic metrics
loss, accuracy = model.evaluate(X_test, y_test_cat)
print(f'Test Accuracy: {accuracy * 100:.2f}%')
print(f'Test Loss: {loss:.4f}')

# Get predictions
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)  # Convert one-hot encoded predictions back to class indices

print("\n===== VALIDATION METRICS =====")

# Confusion Matrix - FIXED: now comparing encoded labels (integers) with predicted integers
conf_matrix = confusion_matrix(y_test_enc, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print("\nClassification Report:")
class_report = classification_report(y_test_enc, y_pred, target_names=class_names)
print(class_report)

# Individual Metrics
print("\nIndividual Metrics:")
print(f"Accuracy: {accuracy_score(y_test_enc, y_pred):.4f}")
print(f"Precision (weighted): {precision_score(y_test_enc, y_pred, average='weighted'):.4f}")
print(f"Recall (weighted): {recall_score(y_test_enc, y_pred, average='weighted'):.4f}")
print(f"F1 Score (weighted): {f1_score(y_test_enc, y_pred, average='weighted'):.4f}")

# Per-class metrics
print("\nPer-class Metrics:")
precision_values = precision_score(y_test_enc, y_pred, average=None)
recall_values = recall_score(y_test_enc, y_pred, average=None)
f1_values = f1_score(y_test_enc, y_pred, average=None)

for i, activity in enumerate(class_names):
    print(f"{activity}:")
    print(f"  Precision: {precision_values[i]:.4f}")
    print(f"  Recall: {recall_values[i]:.4f}")
    print(f"  F1-score: {f1_values[i]:.4f}")

# ROC Curve for multiclass
plt.figure(figsize=(10, 8))
n_classes = len(class_names)
fpr = dict()
tpr = dict()
roc_auc = dict()

# One-vs-Rest approach for multiclass ROC
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve((y_test_enc == i).astype(int),
                                  y_pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of {class_names[i]} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Training history plots
plt.figure(figsize=(12, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Save the model
model.save('/content/drive/MyDrive/T01/activity_recognition_model.h5')
print("Model saved!")
