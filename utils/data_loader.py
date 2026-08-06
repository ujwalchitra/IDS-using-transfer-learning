import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import torch
import warnings
warnings.filterwarnings('ignore')
import os

class IDSDataLoader:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_and_preprocess_data(self, file_path, dataset_type="pretrain"):
        """Load and preprocess dataset"""
        print(f"📁 Loading {dataset_type} data from: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        print(f"✓ Dataset loaded. Shape: {df.shape}")
        
        # Handle different dataset structures
        if dataset_type == "pretrain":
            # For pretrain data: Use 'Label' as target, remove other non-feature columns
            target_col = 'Label'
            columns_to_remove = ['Attack_Class', 'Attack_Class_Encoded']
        else:
            # For finetune data: Use 'Label' as target, remove timestamp
            target_col = 'Label'
            columns_to_remove = ['Timestamp']
        
        # Check if target column exists
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")
        
        # Extract target
        y = df[target_col]
        
        # Create feature dataframe by removing target and other non-feature columns
        feature_df = df.drop(columns=[target_col], errors='ignore')
        
        # Remove additional specified columns
        for col in columns_to_remove:
            if col in feature_df.columns:
                feature_df = feature_df.drop(columns=[col])
                print(f"✓ Removed column: {col}")
        
        # Convert all features to numeric, coercing errors to NaN
        for col in feature_df.columns:
            feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')
        
        # Fill any remaining NaN values with 0
        feature_df = feature_df.fillna(0)
        
        X = feature_df
        
        print(f"✓ Features: {X.shape[1]}, Target: {target_col}")
        
        # Select top features for speed (if needed)
        if X.shape[1] > self.config.INPUT_DIM:
            print(f"⚠️  Too many features ({X.shape[1]}). Selecting top {self.config.INPUT_DIM}...")
            # Use variance-based feature selection
            variances = X.var().sort_values(ascending=False)
            top_features = variances.head(self.config.INPUT_DIM).index
            X = X[top_features]
            print(f"✓ Selected top {self.config.INPUT_DIM} features by variance")
        
        # Encode labels
        print(f"Original labels: {y.unique()}")
        if y.dtype == 'object':
            y_encoded = self.label_encoder.fit_transform(y)
            label_mapping = dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))
            print(f"✓ Labels encoded. Mapping: {label_mapping}")
            
            # For binary classification, map to 0 (Benign) and 1 (Attack)
            if len(np.unique(y_encoded)) > 2:
                print("⚠️  Multi-class detected. Converting to binary: Benign=0, Attack=1")
                # Find which label is Benign
                benign_labels = ['BENIGN', 'Benign', 'benign', 'Normal', 'normal']
                benign_mask = np.zeros_like(y_encoded, dtype=bool)
                
                for benign_label in benign_labels:
                    if benign_label in self.label_encoder.classes_:
                        benign_idx = list(self.label_encoder.classes_).index(benign_label)
                        benign_mask = benign_mask | (y_encoded == benign_idx)
                        break
                
                y_binary = np.where(benign_mask, 0, 1)
                print(f"✓ Binary classes - Benign: {np.sum(y_binary == 0)}, Attack: {np.sum(y_binary == 1)}")
                y = y_binary
            else:
                y = y_encoded
        else:
            print(f"✓ Numeric labels. Classes: {np.unique(y)}")
        
        # Convert to numpy
        X = X.astype(np.float32)
        y = y.astype(np.int64)
        
        print(f"✅ Final dataset shape: {X.shape}")
        print(f"✅ Class distribution: {np.bincount(y)}")
        return X, y
    
    def create_dataloaders(self, X, y, batch_size=None, test_size=0.2):
        """Create train and test dataloaders"""
        if batch_size is None:
            batch_size = self.config.BATCH_SIZE
            
        print(f"🛠️  Creating dataloaders - Batch size: {batch_size}, Test size: {test_size}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"✓ Train set: {X_train.shape}, Test set: {X_test.shape}")
        print(f"✓ Train class distribution: {np.bincount(y_train)}")
        print(f"✓ Test class distribution: {np.bincount(y_test)}")
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Create tensors
        train_dataset = TensorDataset(
            torch.tensor(X_train), 
            torch.tensor(y_train)
        )
        test_dataset = TensorDataset(
            torch.tensor(X_test), 
            torch.tensor(y_test)
        )
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print("✅ DataLoaders created successfully!")
        return train_loader, test_loader

    def test_data_loading(self):
        """Test if data loading works with your datasets"""
        print("🧪 TESTING DATA LOADING...")
        
        try:
            # Test pretrain data
            print("\n" + "="*50)
            print("1. Testing PRETRAIN data...")
            print("="*50)
            X_pretrain, y_pretrain = self.load_and_preprocess_data(
                self.config.PRETRAIN_PATH, "pretrain"
            )
            train_loader, test_loader = self.create_dataloaders(X_pretrain, y_pretrain, batch_size=32)
            
            # Test one batch
            for data, target in train_loader:
                print(f"✓ Pretrain batch - Data: {data.shape}, Targets: {target.shape}")
                print(f"✓ Data range: [{data.min():.2f}, {data.max():.2f}]")
                print(f"✓ Target distribution: {torch.bincount(target)}")
                break
                
            # Test finetune data
            print("\n" + "="*50)
            print("2. Testing FINETUNE data...")
            print("="*50)
            X_finetune, y_finetune = self.load_and_preprocess_data(
                self.config.FINETUNE_PATH, "finetune"
            )
            finetune_loader, _ = self.create_dataloaders(X_finetune, y_finetune, batch_size=32)
            
            for data, target in finetune_loader:
                print(f"✓ Finetune batch - Data: {data.shape}, Targets: {target.shape}")
                print(f"✓ Data range: [{data.min():.2f}, {data.max():.2f}]")
                print(f"✓ Target distribution: {torch.bincount(target)}")
                break
                
            print("\n🎉 DATA LOADING TEST PASSED!")
            return True
            
        except Exception as e:
            print(f"\n❌ DATA LOADING TEST FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False