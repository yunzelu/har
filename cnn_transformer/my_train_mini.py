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
import re

# Configuration class that centralizes all hyperparameters and settings
class Config:
    data_dir = "data/keypoints"                  
    batch_size = 8                         
    num_workers = 0                         
    lr = 0.0003                             
    weight_decay = 1e-5                     
    epochs = 2                             
    num_classes = None                      
    input_dim = 85                          # UPDATED: 68 features (x, y, dx, dy) + 17 scores
    hidden_dim = 256                        
    num_layers = 4                          
    nhead = 8                               
    dropout = 0.4                           
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")  
    save_path = "cnn_transformer/cnn_transformer_model.pth"  
    chunk_size = 64                         
    overlap = 16                            
    max_sequence_length = 536               
    augmentation_prob = 0.7                 
    temporal_smooth_weight = 0.3            
    score_threshold = 0.25                  # NEW: Threshold for masking out joints

# Spatial Attention Module to focus on important keypoints
class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(4, 16),               
            nn.ReLU(),                      
            nn.Linear(16, 1)                # Removed Softmax here to apply the explicit mask first
        )
        
    def forward(self, x, scores):
        # x shape: (batch*seq_len, 17, 4)
        # scores shape: (batch*seq_len, 17)
        
        # 1. HARD MASK: Zero out features of any joint with low confidence
        mask = (scores < Config.score_threshold).unsqueeze(-1)  # Shape: (B, 17, 1)
        x = x.masked_fill(mask, 0.0)
        
        # 2. SOFT ATTENTION: Calculate learned weights from the features
        attn_weights = self.attention(x).squeeze(-1)  # Shape: (B, 17)
        
        # 3. ATTENTION MASK: Force the network to ignore missing joints
        # Setting the weight to -inf ensures Softmax turns it into exactly 0.0
        attn_weights = attn_weights.masked_fill(scores < Config.score_threshold, float('-inf'))
        attn_weights = torch.softmax(attn_weights, dim=1)  
        
        return x * attn_weights.unsqueeze(-1)

# Data augmentation functions
def add_gaussian_noise(seq, std=0.01):
    noisy_seq = seq.clone()
    # UPDATED: Only add noise to the 68 spatial features, keep scores intact
    noisy_seq[:, :68] = seq[:, :68] + torch.randn_like(seq[:, :68]) * std
    return noisy_seq

def random_dropout(seq, p=0.05):
    result = seq.clone()
    # UPDATED: Only dropout spatial features, keep scores intact
    mask = torch.rand_like(seq[:, :68]) > p
    result[:, :68] = seq[:, :68] * mask
    return result

def time_warp(seq, max_warp=0.1):
    batch_size, seq_len, feat_dim = seq.shape
    warp = torch.zeros(batch_size, seq_len, seq_len, device=seq.device)
    for i in range(batch_size):
        warp_factor = 1.0 + (torch.rand(1).item() * 2 - 1) * max_warp  
        for j in range(seq_len):
            pos = int(j * warp_factor)
            if 0 <= pos < seq_len:
                warp[i, j, pos] = 1.0
            else:
                pos = max(0, min(pos, seq_len-1))
                warp[i, j, pos] = 1.0
    warped_seq = torch.bmm(warp, seq)
    return warped_seq

# Dataset class
class KeypointActivityDataset(Dataset):
    def __init__(self, data, labels, config, augment=False):
        self.config = config
        self.data = []
        self.labels = []
        self.augment = augment
        
        label_counts = defaultdict(int)
        for label in labels:
            label_counts[label] += 1
        
        total_samples = len(labels)
        self.class_weights = {cls: total_samples / count for cls, count in label_counts.items()}
        self.sample_weights = []
        
        for sequence, label in zip(data, labels):
            sequence = [frame for frame in sequence if np.any(frame)]  
            chunks = self.chunk_sequence(sequence)
            for chunk in chunks:
                self.data.append(chunk)
                self.labels.append(label)
                self.sample_weights.append(self.class_weights[label])  
                
    def chunk_sequence(self, sequence):
        chunks = []
        seq_len = len(sequence)
        
        if seq_len <= self.config.chunk_size:
            padded = np.zeros((self.config.chunk_size, self.config.input_dim))
            padded[:seq_len] = sequence
            return [padded]
        
        start = 0
        while start < seq_len:
            end = start + self.config.chunk_size
            if end > seq_len:
                end = seq_len
                if seq_len >= self.config.chunk_size:
                    chunk = sequence[seq_len - self.config.chunk_size:seq_len]
                else:
                    chunk = sequence[start:end]
                    if len(chunk) < self.config.chunk_size:
                        padded = np.zeros((self.config.chunk_size, self.config.input_dim))
                        padded[:len(chunk)] = chunk
                        chunk = padded
            else:
                chunk = sequence[start:end]
            
            chunks.append(chunk)
            start += (self.config.chunk_size - self.config.overlap)  
            
            if start + self.config.chunk_size > seq_len:
                break
                
        return chunks
    
    def apply_augmentation(self, sample):
        sample = torch.FloatTensor(sample)
        if self.augment and random.random() < self.config.augmentation_prob:
            seq_len, feat_dim = sample.shape
            sample = sample.unsqueeze(0)
            
            if random.random() < 0.5:
                sample = add_gaussian_noise(sample)
            if random.random() < 0.3:
                sample = random_dropout(sample)
            if random.random() < 0.3:
                sample = time_warp(sample)
            
            sample = sample.squeeze(0)
        return sample
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.apply_augmentation(self.data[idx]) if self.augment else torch.FloatTensor(self.data[idx])
        label = torch.LongTensor([self.labels[idx]])
        return sample, label

# Hybrid CNN-Transformer model
class CNNTransformerModel(nn.Module):
    def __init__(self, config):
        super(CNNTransformerModel, self).__init__()
        self.config = config
        
        self.spatial_attention = SpatialAttention()
        
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),      
            nn.BatchNorm1d(64),                              
            nn.ReLU(),                                       
            nn.MaxPool1d(2),                                 
            nn.Conv1d(64, 128, kernel_size=3, padding=1),    
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, config.hidden_dim, kernel_size=3, padding=1),  
            nn.BatchNorm1d(config.hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(2)                                 
        )
        
        self.pos_encoder = PositionalEncoding(config.hidden_dim, config.dropout)
        
        encoder_layers = TransformerEncoderLayer(
            config.hidden_dim, config.nhead, config.hidden_dim * 4, config.dropout
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, config.num_layers)
        
        self.temporal_attention = nn.Sequential(
            nn.Linear(config.hidden_dim, 1),
            nn.Softmax(dim=0)
        )
        
        self.frame_classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.num_classes)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, 85)
        
        # UPDATED: Split features (68) and scores (17)
        features = x[:, :, :68]
        scores = x[:, :, 68:]
        
        features = features.unsqueeze(1)  
        batch_size, _, seq_len, feat_dim = features.shape
        
        features = features.permute(0, 2, 1, 3)  
        features = features.reshape(batch_size * seq_len, 1, feat_dim)  
        
        # Reshape for spatial attention
        x_flat = features.squeeze(1)  
        x_reshaped = x_flat.view(-1, 17, 4)  
        scores_reshaped = scores.reshape(-1, 17) # Match shape for masking
        
        # Pass BOTH into spatial attention
        x_attn = self.spatial_attention(x_reshaped, scores_reshaped)  
        
        # Reshape back for CNN
        x = x_attn.flatten(1).unsqueeze(1)  
        
        # Extract features with CNN
        x = self.cnn(x)  
        x = x.mean(dim=2)  
        x = x.reshape(batch_size, seq_len, -1)  
        x = x.permute(1, 0, 2)  
        
        x = self.pos_encoder(x)
        x_transformed = self.transformer_encoder(x)  
        
        frame_preds = self.frame_classifier(x_transformed)  
        
        attn_weights = self.temporal_attention(x_transformed)  
        context = torch.sum(x_transformed * attn_weights, dim=0)  
        
        output = self.classifier(context)  
        
        return output, frame_preds

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  
        pe[:, 1::2] = torch.cos(position * div_term)  
        pe = pe.unsqueeze(0).transpose(0, 1)  
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Function to load and preprocess keypoint data
def load_data(data_dir):
    all_data = []
    all_labels = []
    all_subjects = [] # NEW: Keep track of who is in the video
    
    activity_names = sorted(os.listdir(data_dir))
    label_encoder = LabelEncoder()
    
    for activity in tqdm(activity_names, desc="Loading activities"):
        activity_path = os.path.join(data_dir, activity)
        json_files = [f for f in os.listdir(activity_path) if f.endswith('_normalized.json')]
        
        for json_file in json_files:
            # Extract Subject ID using Regex (e.g., finds "17" from "Subject17...")
            match = re.search(r'Subject(\d+)', json_file)
            if match:
                subject_id = int(match.group(1))
            else:
                continue # Skip files if we can't identify the subject
                
            with open(os.path.join(activity_path, json_file), 'r') as f:
                data = json.load(f)
            
            keypoints = []
            scores_list = []
            
            for item in data:
                if isinstance(item, list) and len(item) > 0:
                    frame_dict = item[0]  
                elif isinstance(item, dict):
                    frame_dict = item
                else:
                    continue  
                
                if 'keypoints' in frame_dict and 'scores' in frame_dict:
                    kp = np.array(frame_dict['keypoints']).flatten()
                    score = np.array(frame_dict['scores'])
                    keypoints.append(kp)
                    scores_list.append(score)
            
            if keypoints:
                velocities = []
                prev_kps = None
                prev_scores = None
                
                for kp, current_scores in zip(keypoints, scores_list):
                    if prev_kps is None:
                        velocity = np.zeros_like(kp)
                    else:
                        velocity = kp - prev_kps
                        for i in range(17):
                            if current_scores[i] < Config.score_threshold or prev_scores[i] < Config.score_threshold:
                                velocity[i*2] = 0.0     
                                velocity[i*2 + 1] = 0.0 
                        
                    if not np.isfinite(velocity).all():
                        velocity = np.zeros_like(kp)
                        
                    velocities.append(velocity)
                    prev_kps = kp.copy()
                    prev_scores = current_scores.copy()
                    
                keypoints_with_velocity = [
                    np.concatenate([kp, vel, s]) 
                    for kp, vel, s in zip(keypoints, velocities, scores_list)
                ]
                
                if len(keypoints_with_velocity) > Config.max_sequence_length:
                    keypoints_with_velocity = keypoints_with_velocity[:Config.max_sequence_length]
                
                all_data.append(keypoints_with_velocity)
                all_labels.append(activity)
                all_subjects.append(subject_id) # NEW: Save the subject ID
    
    all_labels = label_encoder.fit_transform(all_labels)
    Config.num_classes = len(label_encoder.classes_)
    
    os.makedirs('cnn_transformer', exist_ok=True)
    np.save('cnn_transformer/cnn_transformer_label_encoder_classes.npy', label_encoder.classes_)
    
    return np.array(all_data, dtype=object), np.array(all_labels), np.array(all_subjects), label_encoder

# Training and Evaluation Functions
def train(model, train_loader, val_loader, config, test_loader=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(config.save_path), exist_ok=True)
    os.makedirs('cnn_transformer/results', exist_ok=True)
    
    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}"):
            inputs = inputs.to(config.device)
            labels = labels.to(config.device).squeeze()
            
            optimizer.zero_grad()
            seq_outputs, frame_outputs = model(inputs)
            seq_loss = criterion(seq_outputs, labels)
            
            expanded_labels = labels.unsqueeze(0).expand(frame_outputs.size(0), -1)
            frame_loss = F.cross_entropy(
                frame_outputs.view(-1, config.num_classes),
                expanded_labels.contiguous().view(-1)
            )
            
            loss = seq_loss + config.temporal_smooth_weight * frame_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        val_loss, val_acc = evaluate(model, val_loader, config)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.save_path)
            print(f"Saved new best model with val_loss: {val_loss:.4f}")
        
        print(f"Epoch {epoch+1}/{config.epochs} - "
              f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - "
              f"Val Acc: {val_acc:.4f} - LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if test_loader and (epoch + 1) % 5 == 0:
            test_loss, test_acc = evaluate(model, test_loader, config)
            print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
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
    
    print("Training complete. Evaluating final model...")
    model.load_state_dict(torch.load(config.save_path))
    
    val_loss, val_acc = evaluate(model, val_loader, config)
    print(f"Final Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
    
    if test_loader:
        test_loss, test_acc = evaluate(model, test_loader, config)
        print(f"Final Test - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
        
        confusion = evaluate_with_confusion(model, test_loader, config)
        np.save('cnn_transformer/results/confusion_matrix.npy', confusion)
        return model, confusion
    
    return model

def evaluate(model, loader, config):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(config.device)
            labels = labels.to(config.device).squeeze()
            
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    loss = running_loss / len(loader.dataset)
    accuracy = correct / total
    return loss, accuracy

def evaluate_with_confusion(model, loader, config):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(config.device)
            labels = labels.to(config.device).squeeze()
            
            outputs, _ = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    confusion = np.zeros((config.num_classes, config.num_classes))
    for pred, label in zip(all_preds, all_labels):
        confusion[label, pred] += 1
        
    row_sums = confusion.sum(axis=1, keepdims=True)
    # Prevent division by zero
    row_sums[row_sums == 0] = 1 
    normalized_confusion = confusion / row_sums
    return normalized_confusion

def calculate_metrics(model, data_loader, config, phase="Validation"):
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(config.device)
            labels = labels.to(config.device).squeeze()
            
            outputs, _ = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(F.softmax(outputs, dim=1).cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    accuracy = np.mean(all_preds == all_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)
    
    try:
        class_names = np.load('cnn_transformer/cnn_transformer_label_encoder_classes.npy')
    except:
        class_names = [f"Class {i}" for i in range(len(class_precision))]
        
    cm = confusion_matrix(all_labels, all_preds)
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    row_sums[row_sums == 0] = 1 # Avoid division by zero
    cm_normalized = cm.astype('float') / row_sums
    
    print(f"\n{phase} Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall: {recall:.4f}")
    print(f"Weighted F1-score: {f1:.4f}")
    
    print("\nPer-class metrics:")
    for i, (p, r, f) in enumerate(zip(class_precision, class_recall, class_f1)):
        print(f"{class_names[i]}: Precision={p:.4f}, Recall={r:.4f}, F1={f:.4f}")
        
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{phase} Confusion Matrix (Normalized)')
    plt.tight_layout()
    plt.savefig(f'cnn_transformer/results/{phase.lower()}_confusion_matrix.png')
    plt.close()
    
    report = classification_report(all_labels, all_preds, target_names=class_names, labels=np.arange(len(class_names)), zero_division=0)
    print("\nClassification Report:")
    print(report)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': cm.tolist(),
        'per_class_precision': class_precision.tolist(),
        'per_class_recall': class_recall.tolist(),
        'per_class_f1': class_f1.tolist(),
    }
    
    np.save(f'cnn_transformer/results/{phase.lower()}_metrics.npy', metrics)
    return metrics

# Main function
def main():
    config = Config()
    print(f"Using device: {config.device}")
    
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        
    print("Loading data...")
    # Receive the subject list from our updated load_data
    X, y, subjects, label_encoder = load_data(config.data_dir)

    classes = label_encoder.classes_
    class_counts = np.bincount(y)
    for i, (cls, count) in enumerate(zip(classes, class_counts)):
        print(f"Class {i} - {cls}: {count} samples")
        
    # ==========================================
    # NEW: 4-Way Subject-Wise Split
    # ==========================================
    unique_subjects = np.unique(subjects)
    print(f"Total Unique Subjects found: {len(unique_subjects)}")
    
    # Shuffle subjects to ensure random distribution
    np.random.shuffle(unique_subjects)
    
    # Manually define the split (10 Train, 2 Val, 2 Calib, 3 Test)
    train_subs = unique_subjects[9:10]
    val_subs = unique_subjects[10:12]
    calib_subs = unique_subjects[12:14]
    test_subs = unique_subjects[14:]
    
    print(f"Train Subjects: {train_subs}")
    print(f"Val Subjects: {val_subs}")
    print(f"Calibration Subjects: {calib_subs}")
    print(f"Test Subjects: {test_subs}")
    
    # Helper function to extract data based on subject lists
    def filter_by_subject(data, labels, sub_array, target_subs):
        mask = np.isin(sub_array, target_subs)
        # Using list comprehension to safely extract the variable-length sequences
        filtered_X = [data[i] for i in range(len(data)) if mask[i]]
        filtered_y = labels[mask]
        return filtered_X, filtered_y

    # Create the actual splits
    X_train, y_train = filter_by_subject(X, y, subjects, train_subs)
    X_val, y_val = filter_by_subject(X, y, subjects, val_subs)
    X_calib, y_calib = filter_by_subject(X, y, subjects, calib_subs)
    X_test, y_test = filter_by_subject(X, y, subjects, test_subs)
    # ==========================================
    
    print(f"Train: {len(X_train)} samples")
    print(f"Validation: {len(X_val)} samples")
    print(f"Calibration: {len(X_calib)} samples (Reserved for Conformal Prediction)")
    print(f"Test: {len(X_test)} samples")
    
    train_dataset = KeypointActivityDataset(X_train, y_train, config, augment=True)
    val_dataset = KeypointActivityDataset(X_val, y_val, config, augment=False)
    test_dataset = KeypointActivityDataset(X_test, y_test, config, augment=False)
    # Note: We don't load the Calibration set into a DataLoader right now, 
    # you just keep X_calib, y_calib saved for your future script.
    
    # Save the calibration data for future conformal prediction step
    np.save('cnn_transformer/calibration_X.npy', np.array(X_calib, dtype=object))
    np.save('cnn_transformer/calibration_y.npy', y_calib)
    
    weights = torch.DoubleTensor(train_dataset.sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=sampler, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    
    model = CNNTransformerModel(config).to(config.device)
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    
    print("Starting training...")
    model, confusion = train(model, train_loader, val_loader, config, test_loader)

    print("\nCalculating validation metrics...")
    val_metrics = calculate_metrics(model, val_loader, config, "Validation")
    
    print("\nCalculating test metrics...")
    test_metrics = calculate_metrics(model, test_loader, config, "Test")
    
    model_info = {
        'config': {k: v for k, v in config.__dict__.items() if not k.startswith('__') and not callable(v) and k != 'device'},
        'classes': list(label_encoder.classes_),
        'label_encoder_path': 'cnn_transformer_label_encoder_classes.npy',
        'model_path': config.save_path,
        'input_dim': config.input_dim,
        'num_classes': config.num_classes,
    }
    
    with open('cnn_transformer/model_info.json', 'w') as f:
        serializable_info = {}
        for k, v in model_info.items():
            if isinstance(v, dict):
                serializable_info[k] = {kk: str(vv) if not isinstance(vv, (int, float, str, list, dict, bool, type(None))) else vv 
                                       for kk, vv in v.items()}
            else:
                serializable_info[k] = str(v) if not isinstance(v, (int, float, str, list, dict, bool, type(None))) else v
        json.dump(serializable_info, f, indent=2)
    
    print("Training completed successfully!")
    print(f"Model saved to {config.save_path}")
    print(f"Model info saved to cnn_transformer/model_info.json")

if __name__ == "__main__":
    main()