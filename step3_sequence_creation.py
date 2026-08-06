# IoT Intrusion Detection - Step 3: Sequence Creation for 1D-CNN + LSTM
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

print("🚀 IoT Intrusion Detection - Step 3: Sequence Creation for 1D-CNN + LSTM")
print("=" * 70)

# ============================================================================
# STEP 3.1: LOAD PROCESSED DATA (FIXED)
# ============================================================================
print("\n📥 LOADING PROCESSED DATA...")

base_path = "D:/IoT-DDoS-Detection/iot_ddos_detection/"
processed_data_path = os.path.join(base_path, "02_processed_data")
preprocessing_path = os.path.join(base_path, "06_preprocessing")

# FIX: Load with allow_pickle=True
processed_file = os.path.join(processed_data_path, "processed_features.npz")
data = np.load(processed_file, allow_pickle=True)

X = data['X']
y = data['y']
feature_names = data['feature_names']
class_names = data['class_names']

print(f"✅ Loaded processed data:")
print(f"   Features (X): {X.shape}")
print(f"   Targets (y): {y.shape}")
print(f"   Number of features: {len(feature_names)}")
print(f"   Classes: {list(class_names)}")

# Load preprocessing objects
preprocessor_file = os.path.join(preprocessing_path, "preprocessing_objects.pkl")
preprocessor = joblib.load(preprocessor_file)
scaler = preprocessor['scaler']
label_encoder = preprocessor['label_encoder']
feature_columns = preprocessor['feature_columns']

print(f"✅ Loaded preprocessing objects")

# ============================================================================
# STEP 3.2: CREATE TIME-SERIES SEQUENCES
# ============================================================================
print("\n🔄 CREATING TIME-SERIES SEQUENCES...")

def create_sequences(features, targets, sequence_length=50, step_size=10):
    """
    Create sequences for 1D-CNN + LSTM model
    Each sequence contains 'sequence_length' consecutive network flows
    """
    sequences = []
    sequence_labels = []
    
    # Create sliding window sequences
    for i in range(0, len(features) - sequence_length, step_size):
        sequence = features[i:i + sequence_length]
        label = targets[i + sequence_length - 1]  # Use last flow's label
        
        sequences.append(sequence)
        sequence_labels.append(label)
    
    return np.array(sequences), np.array(sequence_labels)

# Parameters for sequence creation
SEQUENCE_LENGTH = 50  # Number of consecutive flows in each sequence
STEP_SIZE = 10        # Step between sequences (1 = dense, 10 = sparse)

print(f"   Sequence length: {SEQUENCE_LENGTH} flows")
print(f"   Step size: {STEP_SIZE}")
print(f"   Creating sequences...")

X_sequences, y_sequences = create_sequences(X, y, 
                                          sequence_length=SEQUENCE_LENGTH, 
                                          step_size=STEP_SIZE)

print(f"✅ Sequences created:")
print(f"   Input sequences: {X_sequences.shape}")  # (samples, sequence_length, features)
print(f"   Output labels: {y_sequences.shape}")

# ============================================================================
# STEP 3.3: SPLIT DATA INTO TRAIN/VALIDATION/TEST SETS
# ============================================================================
print("\n📊 SPLITTING DATA INTO TRAIN/VALIDATION/TEST SETS...")

# First split: 80% train, 20% temp (validation + test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X_sequences, y_sequences, test_size=0.2, random_state=42, stratify=y_sequences
)

# Second split: 50% temp → 10% validation, 10% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"✅ Data split completed:")
print(f"   Training set:   {X_train.shape} - {len(y_train):,} sequences")
print(f"   Validation set: {X_val.shape} - {len(y_val):,} sequences") 
print(f"   Test set:       {X_test.shape} - {len(y_test):,} sequences")

# Check class distribution
def print_class_distribution(y_data, set_name):
    unique, counts = np.unique(y_data, return_counts=True)
    print(f"   {set_name:12} ", end="")
    for cls, count in zip(unique, counts):
        percentage = (count / len(y_data)) * 100
        print(f"{class_names[cls]}: {count:,} ({percentage:.1f}%) ", end="")
    print()

print(f"\n🎯 CLASS DISTRIBUTION:")
print_class_distribution(y_train, "Training")
print_class_distribution(y_val, "Validation")
print_class_distribution(y_test, "Test")

# ============================================================================
# STEP 3.4: SAVE SEQUENCE DATA
# ============================================================================
print("\n💾 SAVING SEQUENCE DATA...")

sequence_data_path = os.path.join(base_path, "03_sequence_data")
os.makedirs(sequence_data_path, exist_ok=True)

# Save sequence datasets
sequence_file = os.path.join(sequence_data_path, "sequence_datasets.npz")
np.savez(sequence_file,
         X_train=X_train, y_train=y_train,
         X_val=X_val, y_val=y_val, 
         X_test=X_test, y_test=y_test,
         feature_names=feature_names,
         class_names=class_names,
         sequence_length=SEQUENCE_LENGTH)

print(f"✅ Sequence datasets: 03_sequence_data/sequence_datasets.npz")

# Save sequence parameters
params_file = os.path.join(sequence_data_path, "sequence_parameters.pkl")
joblib.dump({
    'sequence_length': SEQUENCE_LENGTH,
    'step_size': STEP_SIZE,
    'input_shape': (SEQUENCE_LENGTH, len(feature_names)),
    'num_classes': len(class_names)
}, params_file)

print(f"✅ Sequence parameters: 03_sequence_data/sequence_parameters.pkl")

# ============================================================================
# STEP 3.5: FINAL SUMMARY
# ============================================================================
print("\n🎉 STEP 3 COMPLETED SUCCESSFULLY!")
print("=" * 50)

print(f"📊 SEQUENCE DATA SUMMARY:")
print(f"   Input shape: ({SEQUENCE_LENGTH}, {len(feature_names)})")
print(f"   Training sequences:   {len(X_train):,}")
print(f"   Validation sequences: {len(X_val):,}")
print(f"   Test sequences:       {len(X_test):,}")
print(f"   Total sequences:      {len(X_sequences):,}")
print(f"   Number of classes:    {len(class_names)}")

print(f"\n🔜 NEXT STEP: Build 1D-CNN + LSTM Hybrid Model")
print("   We'll create the neural network that learns from these sequences!")

print(f"\n📁 FILES CREATED:")
print(f"   03_sequence_data/sequence_datasets.npz - All sequence data")
print(f"   03_sequence_data/sequence_parameters.pkl - Model parameters")