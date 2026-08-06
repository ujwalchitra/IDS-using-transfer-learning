# Try to import torch, if not available, provide installation instructions
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not found. Please install it using:")
    print("pip install torch torchvision torchaudio")
    exit(1)

import os

class Config:
    # Data paths
# config.py - UPDATE THESE PATHS:
    # config.py - UPDATE TO SHUFFLED FILES:
    PRETRAIN_PATH = r"D:\IoT-DDoS-Detection\02_processed_data\cleaned_intrusion_dataset_SHUFFLED.csv"
    FINETUNE_PATH = r"D:\IoT-DDoS-Detection\02_processed_data\ut2024_finetune_sample_SHUFFLED.csv"
    
    # Model parameters - Optimized for speed
    PRETRAIN_EPOCHS = 20
    FINETUNE_EPOCHS = 15
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    NUM_CLASSES = 2
    
    # Lightweight architecture - Updated based on your data
    INPUT_DIM = 50  # Your data has 70 features, we'll use top 50
    HIDDEN_DIM = 64
    CNN_FILTERS = 32
    KERNEL_SIZE = 3
    LSTM_LAYERS = 1
    DROPOUT = 0.2
    EMBEDDING_DIM = 16
    
    # Training
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EARLY_STOPPING_PATIENCE = 5
    
config = Config()

# Print device information
print(f"Using device: {config.DEVICE}")
if config.DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")