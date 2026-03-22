import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import random
import torch.nn.functional as F
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
import seaborn as sns

# Configuration class that centralizes all hyperparameters and settings
class Config:
    data_dir = "data\keypoints"                  # Directory containing the keypoint data
    batch_size = 64                         # Number of samples per batch (reduced for better generalization)
    num_workers = 4                         # Number of parallel workers for data loading
    lr = 0.0003                             # Learning rate for optimizer
    weight_decay = 1e-5                     # L2 regularization to prevent overfitting
    epochs = 12                             # Number of training epochs
    num_classes = None                      # Will be determined from the dataset
    input_dim = 68                          # 17 keypoints with (x, y, dx, dy) values per keypoint
    hidden_dim = 256                        # Size of hidden layers in the network
    num_layers = 4                          # Number of transformer encoder layers
    nhead = 8                               # Number of attention heads in transformer
    dropout = 0.4                           # Dropout rate for regularization
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")  # Auto-select device
    save_path = "cnn_transformer/cnn_transformer_model.pth"  # Path to save the trained model
    chunk_size = 64                         # Size of sequence chunks for processing
    overlap = 16                            # Overlap between consecutive chunks
    max_sequence_length = 536               # Maximum allowed sequence length
    augmentation_prob = 0.7                 # Probability of applying data augmentation
    temporal_smooth_weight = 0.3            # Weight for temporal smoothness loss component

# Spatial Attention Module to focus on important keypoints
class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # MLP to generate attention weights for each keypoint
        self.attention = nn.Sequential(
            nn.Linear(4, 16),               # Map each keypoint's features to higher dimension
            nn.ReLU(),                      # Non-linearity
            nn.Linear(16, 1),               # Compress to a single attention score
            nn.Softmax(dim=1)               # Normalize attention weights across keypoints
        )
        
    def forward(self, x):
        # x shape: (batch*seq_len, 17, 4) - batch of sequences with 17 keypoints, each with 4 values
        attn_weights = self.attention(x).squeeze(-1)  # Calculate attention weights
        attn_weights = torch.softmax(attn_weights, dim=1)  # Ensure weights sum to 1
        return x * attn_weights.unsqueeze(-1)  # Apply attention weights to input features

# Data augmentation functions to increase dataset variability and improve generalization

# Add random Gaussian noise to keypoints
def add_gaussian_noise(seq, std=0.01):
    return seq + torch.randn_like(seq) * std

# Randomly set some keypoints to zero to simulate occlusion
def random_dropout(seq, p=0.05):
    mask = torch.rand_like(seq) > p
    return seq * mask

# Apply time warping to simulate variations in movement speed
def time_warp(seq, max_warp=0.1):
    batch_size, seq_len, feat_dim = seq.shape
    
    # Create time warping matrix
    warp = torch.zeros(batch_size, seq_len, seq_len, device=seq.device)
    for i in range(batch_size):
        warp_factor = 1.0 + (torch.rand(1).item() * 2 - 1) * max_warp  # Random factor between 1-max_warp and 1+max_warp
        for j in range(seq_len):
            # Calculate warped position
            pos = int(j * warp_factor)
            if 0 <= pos < seq_len:
                warp[i, j, pos] = 1.0
            else:
                # Clamp to valid position if outside bounds
                pos = max(0, min(pos, seq_len-1))
                warp[i, j, pos] = 1.0
    
    # Apply warping through matrix multiplication
    warped_seq = torch.bmm(warp, seq)
    return warped_seq

# Dataset class that handles chunking, augmentation, and class weighting
class KeypointActivityDataset(Dataset):
    def __init__(self, data, labels, config, augment=False):
        self.config = config
        self.data = []
        self.labels = []
        self.augment = augment
        
        # Calculate class weights to handle imbalanced data
        label_counts = defaultdict(int)
        for label in labels:
            label_counts[label] += 1
        
        # Inverse frequency weighting: less frequent classes get higher weights
        total_samples = len(labels)
        self.class_weights = {cls: total_samples / count for cls, count in label_counts.items()}
        self.sample_weights = []
        
        # Process each sequence into chunks
        for sequence, label in zip(data, labels):
            sequence = [frame for frame in sequence if np.any(frame)]  # Remove empty frames
            chunks = self.chunk_sequence(sequence)
            for chunk in chunks:
                self.data.append(chunk)
                self.labels.append(label)
                self.sample_weights.append(self.class_weights[label])  # Store weight for each sample
                
    # Function to chunk sequences into smaller overlapping segments            
    def chunk_sequence(self, sequence):
        chunks = []
        seq_len = len(sequence)
        
        # If sequence is shorter than chunk size, pad it
        if seq_len <= self.config.chunk_size:
            padded = np.zeros((self.config.chunk_size, self.config.input_dim))
            padded[:seq_len] = sequence
            return [padded]
        
        # Create overlapping chunks for longer sequences
        start = 0
        while start < seq_len:
            end = start + self.config.chunk_size
            if end > seq_len:
                end = seq_len
                # Create a chunk with the last config.chunk_size frames
                if seq_len >= self.config.chunk_size:
                    chunk = sequence[seq_len - self.config.chunk_size:seq_len]
                else:
                    chunk = sequence[start:end]
                    # Pad if needed
                    if len(chunk) < self.config.chunk_size:
                        padded = np.zeros((self.config.chunk_size, self.config.input_dim))
                        padded[:len(chunk)] = chunk
                        chunk = padded
            else:
                chunk = sequence[start:end]
            
            chunks.append(chunk)
            start += (self.config.chunk_size - self.config.overlap)  # Move forward with overlap
            
            # Don't create chunks that would be mostly padding
            if start + self.config.chunk_size > seq_len:
                break
                
        return chunks
    
    # Function for applying data augmentation
    def apply_augmentation(self, sample):
        sample = torch.FloatTensor(sample)
        
        if self.augment and random.random() < self.config.augmentation_prob:
            # Reshape to (1, seq_len, dim) for time warp
            seq_len, feat_dim = sample.shape
            sample = sample.unsqueeze(0)
            
            # Apply augmentations with different probabilities
            if random.random() < 0.5:
                sample = add_gaussian_noise(sample)
            
            if random.random() < 0.3:
                sample = random_dropout(sample)
            
            if random.random() < 0.3:
                sample = time_warp(sample)
            
            # Convert back to (seq_len, dim)
            sample = sample.squeeze(0)
        
        return sample
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Apply augmentation if enabled, then convert to tensor
        sample = self.apply_augmentation(self.data[idx]) if self.augment else torch.FloatTensor(self.data[idx])
        label = torch.LongTensor([self.labels[idx]])
        return sample, label

# Hybrid CNN-Transformer model for activity recognition with temporal consistency
class CNNTransformerModel(nn.Module):
    def __init__(self, config):
        super(CNNTransformerModel, self).__init__()
        self.config = config
        
        # Spatial Attention to focus on important keypoints
        self.spatial_attention = SpatialAttention()
        
        # CNN for feature extraction from keypoint data
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),      # First conv layer
            nn.BatchNorm1d(64),                              # Batch normalization
            nn.ReLU(),                                       # Activation
            nn.MaxPool1d(2),                                 # Pooling to reduce dimensions
            nn.Conv1d(64, 128, kernel_size=3, padding=1),    # Second conv layer
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, config.hidden_dim, kernel_size=3, padding=1),  # Third conv layer
            nn.BatchNorm1d(config.hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(2)                                 # Final feature map
        )
        
        # Positional encoding for transformer
        self.pos_encoder = PositionalEncoding(config.hidden_dim, config.dropout)
        
        # Transformer encoder for modeling temporal relationships
        encoder_layers = TransformerEncoderLayer(
            config.hidden_dim, config.nhead, config.hidden_dim * 4, config.dropout
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, config.num_layers)
        
        # Temporal attention to focus on important frames
        self.temporal_attention = nn.Sequential(
            nn.Linear(config.hidden_dim, 1),
            nn.Softmax(dim=0)
        )
        
        # Frame-level classifier for temporal consistency loss
        self.frame_classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.num_classes)
        )
        
        # Sequence-level classifier for final prediction
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = x.unsqueeze(1)  # Add channel dimension for CNN
        batch_size, _, seq_len, input_dim = x.shape
        x = x.permute(0, 2, 1, 3)  # Reshape to (batch_size, seq_len, 1, input_dim)
        x = x.reshape(batch_size * seq_len, 1, input_dim)  # Flatten batch and seq dimensions
        
        # Apply spatial attention to focus on important keypoints
        x_flat = x.squeeze(1)  # (batch*seq_len, 68)
        x_reshaped = x_flat.view(-1, 17, 4)  # Reshape to (batch*seq_len, 17, 4) for keypoint processing
        x_attn = self.spatial_attention(x_reshaped)  # Apply attention
        x = x_attn.flatten(1).unsqueeze(1)  # Reshape back for CNN
        
        # Extract features with CNN
        x = self.cnn(x)  # (batch*seq_len, hidden_dim, feature_dim)
        x = x.mean(dim=2)  # Global average pooling
        x = x.reshape(batch_size, seq_len, -1)  # Restore batch and sequence dimensions
        x = x.permute(1, 0, 2)  # Reorder to (seq_len, batch_size, hidden_dim) for transformer
        
        # Apply positional encoding for transformer
        x = self.pos_encoder(x)
        
        # Process sequence with transformer to model temporal relationships
        x_transformed = self.transformer_encoder(x)  # (seq_len, batch_size, hidden_dim)
        
        # Generate frame-level predictions for temporal consistency
        frame_preds = self.frame_classifier(x_transformed)  # (seq_len, batch_size, num_classes)
        
        # Apply temporal attention to focus on key frames
        attn_weights = self.temporal_attention(x_transformed)  # (seq_len, batch_size, 1)
        context = torch.sum(x_transformed * attn_weights, dim=0)  # Weighted sum
        
        # Final sequence-level classification
        output = self.classifier(context)  # (batch_size, num_classes)
        
        return output, frame_preds

# Positional Encoding for transformer to encode position information
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create a matrix of position encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)  # Even positions
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd positions
        pe = pe.unsqueeze(0).transpose(0, 1)  # Reshape for addition
        
        # Register buffer (not a parameter but should be part of model state)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Add positional encoding to input
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Function to load and preprocess keypoint data with velocity features
def load_data(data_dir):
    all_data = []
    all_labels = []
    activity_names = sorted(os.listdir(data_dir))
    label_encoder = LabelEncoder()
    
    # Process each activity folder
    for activity in tqdm(activity_names, desc="Loading activities"):
        activity_path = os.path.join(data_dir, activity)
        json_files = [f for f in os.listdir(activity_path) if f.endswith('_normalized.json')]
        
        for json_file in json_files:
            with open(os.path.join(activity_path, json_file), 'r') as f:
                data = json.load(f)
            
            for sequence in data:
                keypoints = []
                for frame in sequence:
                    if 'keypoints' in frame:
                        kp = np.array(frame['keypoints']).flatten()
                        keypoints.append(kp)
                
                if keypoints:
                    # Calculate velocities exactly as in inference
                    # This ensures consistency between training and inference
                    velocities = []
                    prev_kps = None
                    
                    for kp in keypoints:
                        if prev_kps is None:
                            # First frame - zero velocity
                            velocity = np.zeros_like(kp)
                        else:
                            # Calculate velocity as frame-to-frame difference
                            velocity = kp - prev_kps
                            
                        # Handle numerical instabilities
                        if not np.isfinite(velocity).all():
                            velocity = np.zeros_like(kp)
                            
                        velocities.append(velocity)
                        prev_kps = kp.copy()
                        
                    # Concatenate keypoints and velocities for each frame
                    keypoints_with_velocity = [
                        np.concatenate([kp, vel]) 
                        for kp, vel in zip(keypoints, velocities)
                    ]
                    
                    # Truncate very long sequences
                    if len(keypoints_with_velocity) > Config.max_sequence_length:
                        keypoints_with_velocity = keypoints_with_velocity[:Config.max_sequence_length]
                    
                    all_data.append(keypoints_with_velocity)
                    all_labels.append(activity)
    
    # Encode activity labels as integers
    all_labels = label_encoder.fit_transform(all_labels)
    Config.num_classes = len(label_encoder.classes_)
    
    # Save label encoder classes for inference
    np.save('cnn_transformer/cnn_transformer_label_encoder_classes.npy', label_encoder.classes_)
    
    return all_data, all_labels, label_encoder

# Training function with temporal consistency loss
def train(model, train_loader, val_loader, config, test_loader=None):
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    # Learning rate scheduler that reduces LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}"):
            inputs = inputs.to(config.device)
            labels = labels.to(config.device).squeeze()
            
            optimizer.zero_grad()
            
            # Forward pass returns both sequence-level and frame-level predictions
            seq_outputs, frame_outputs = model(inputs)
            
            # Sequence classification loss (main objective)
            seq_loss = criterion(seq_outputs, labels)
            
            # Frame-level classification loss for temporal consistency
            # We expand labels to match frame_outputs shape since each frame should predict the sequence label
            expanded_labels = labels.unsqueeze(0).expand(frame_outputs.size(0), -1)
            frame_loss = F.cross_entropy(
                frame_outputs.view(-1, config.num_classes),
                expanded_labels.contiguous().view(-1)
            )
            
            # Combined loss with temporal smoothness regularization
            # This encourages consistent predictions across the sequence
            loss = seq_loss + config.temporal_smooth_weight * frame_loss
            
            loss.backward()
            # Gradient clipping to stabilize training
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        # Calculate average training loss
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Evaluate on validation set
        val_loss, val_acc = evaluate(model, val_loader, config)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Save model if it's the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.save_path)
            print(f"Saved new best model with val_loss: {val_loss:.4f}")
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{config.epochs} - "
              f"Train Loss: {train_loss:.4f} - "
              f"Val Loss: {val_loss:.4f} - "
              f"Val Acc: {val_acc:.4f} - "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Periodically evaluate on test set if available
        if test_loader and (epoch + 1) % 5 == 0:
            test_loss, test_acc = evaluate(model, test_loader, config)
            print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    # Plot training curves for visualization
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')
    
    plt.savefig('cnn_transformer/results/training_curves.png')
    plt.close()
    
    # Final evaluation using best model
    print("Training complete. Evaluating final model...")
    model.load_state_dict(torch.load(config.save_path))
    
    val_loss, val_acc = evaluate(model, val_loader, config)
    print(f"Final Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
    
    if test_loader:
        test_loss, test_acc = evaluate(model, test_loader, config)
        print(f"Final Test - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
        
        # Generate confusion matrix for detailed error analysis
        confusion = evaluate_with_confusion(model, test_loader, config)
        np.save('cnn_transformer/results/confusion_matrix.npy', confusion)
        
        return model, confusion
    
    return model

# Evaluation function for calculating loss and accuracy
def evaluate(model, loader, config):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # No gradients needed for evaluation
        for inputs, labels in loader:
            inputs = inputs.to(config.device)
            labels = labels.to(config.device).squeeze()
            
            # Forward pass (only using sequence output for evaluation)
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Calculate average loss and accuracy
    loss = running_loss / len(loader.dataset)
    accuracy = correct / total
    return loss, accuracy

# Function to compute and save confusion matrix for error analysis
def evaluate_with_confusion(model, loader, config):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(config.device)
            labels = labels.to(config.device).squeeze()
            
            # Forward pass
            outputs, _ = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            # Collect all predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute confusion matrix
    confusion = np.zeros((config.num_classes, config.num_classes))
    for pred, label in zip(all_preds, all_labels):
        confusion[label, pred] += 1
    
    # Normalize by row (true label) to get per-class accuracy
    row_sums = confusion.sum(axis=1, keepdims=True)
    normalized_confusion = confusion / row_sums
    
    return normalized_confusion

# Function to calculate comprehensive metrics for model evaluation
def calculate_metrics(model, data_loader, config, phase="Validation"):
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(config.device)
            labels = labels.to(config.device).squeeze()
            
            # Forward pass
            outputs, _ = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            # Collect predictions and true labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(F.softmax(outputs, dim=1).cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    # Calculate metrics
    accuracy = np.mean(all_preds == all_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    # Calculate per-class metrics
    class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None
    )
    
    # Load class names
    try:
        class_names = np.load('cnn_transformer_label_encoder_classes.npy')
    except:
        class_names = [f"Class {i}" for i in range(len(class_precision))]
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Print metrics
    print(f"\n{phase} Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall: {recall:.4f}")
    print(f"Weighted F1-score: {f1:.4f}")
    
    print("\nPer-class metrics:")
    for i, (p, r, f) in enumerate(zip(class_precision, class_recall, class_f1)):
        print(f"{class_names[i]}: Precision={p:.4f}, Recall={r:.4f}, F1={f:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{phase} Confusion Matrix (Normalized)')
    plt.tight_layout()
    plt.savefig(f'cnn_transformer/results/{phase.lower()}_confusion_matrix.png')
    plt.close()
    
    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'per_class_precision': class_precision,
        'per_class_recall': class_recall,
        'per_class_f1': class_f1,
    }
    
    # Save metrics to file
    np.save(f'cnn_transformer/results/{phase.lower()}_metrics.npy', metrics)
    
    return metrics

# Main function to orchestrate the training process
def main():
    config = Config()
    print(f"Using device: {config.device}")
    
    # Set random seeds for reproducibility across runs
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Load and preprocess data
    print("Loading data...")
    X, y, label_encoder = load_data(config.data_dir)
    
    # Print class distribution to check for imbalance
    classes = label_encoder.classes_
    class_counts = np.bincount(y)
    for i, (cls, count) in enumerate(zip(classes, class_counts)):
        print(f"Class {i} - {cls}: {count} samples")
    
    # Split data into train, validation, and test sets
    # Stratified split maintains class proportions
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
    )
    
    print(f"Train: {len(X_train)} samples")
    print(f"Validation: {len(X_val)} samples")
    print(f"Test: {len(X_test)} samples")
    
    # Create datasets with appropriate augmentation settings
    train_dataset = KeypointActivityDataset(X_train, y_train, config, augment=True)
    val_dataset = KeypointActivityDataset(X_val, y_val, config, augment=False)
    test_dataset = KeypointActivityDataset(X_test, y_test, config, augment=False)
    
    # Create weighted sampler for training to handle class imbalance
    # This ensures balanced sampling of classes during training
    weights = torch.DoubleTensor(train_dataset.sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights, len(weights), replacement=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, 
        sampler=sampler, num_workers=config.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, 
        shuffle=True, num_workers=config.num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, 
        shuffle=False, num_workers=config.num_workers
    )
    
    # Initialize model
    model = CNNTransformerModel(config).to(config.device)
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    
    # Train model
    print("Starting training...")
    model, confusion = train(model, train_loader, val_loader, config, test_loader)

    # Calculate and save comprehensive metrics
    print("\nCalculating validation metrics...")
    val_metrics = calculate_metrics(model, val_loader, config, "Validation")
    
    print("\nCalculating test metrics...")
    test_metrics = calculate_metrics(model, test_loader, config, "Test")
    
    # Save model metadata and label encoder for inference
    model_info = {
        'config': {k: v for k, v in config.__dict__.items() if not k.startswith('__') and not callable(v) and k != 'device'},
        'classes': list(label_encoder.classes_),
        'label_encoder_path': 'label_encoder_classes.npy',
        'model_path': config.save_path,
        'input_dim': config.input_dim,
        'num_classes': config.num_classes,
    }
    
    # Save model info to JSON file, handling non-serializable types
    with open('cnn_transformer/model_info.json', 'w') as f:
        # Convert any non-serializable values to strings
        serializable_info = {}
        for k, v in model_info.items():
            if isinstance(v, dict):
                serializable_info[k] = {kk: str(vv) if not isinstance(vv, (int, float, str, list, dict, bool, type(None))) else vv 
                                       for kk, vv in v.items()}
            else:
                serializable_info[k] = str(v) if not isinstance(v, (int, float, str, list, dict, bool, type(None))) else v
        
        json.dump(serializable_info, f, indent=2)
    
    # Print summary of training results
    print("Training completed successfully!")
    print(f"Model saved to {config.save_path}")
    print(f"Model info saved to model_info.json")
    print(f"Label encoder saved to {model_info['label_encoder_path']}")

if __name__ == "__main__":
    main()