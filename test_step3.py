# test_step3.py
import torch
import torch.nn as nn
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("🧪 TESTING STEP 3: Hybrid 1D-CNN + LSTM Model")

try:
    from config import Config
    from models.hybrid_1dcnn_lstm import Hybrid1DCNN_LSTM
    
    config = Config()
    print("✓ Config loaded")
    
    # Create model
    model = Hybrid1DCNN_LSTM(config).to(config.DEVICE)
    print("✓ Model created successfully")
    
    # Test with sample data
    print("🧪 Testing model with sample data...")
    batch_size = 32
    sample_input = torch.randn(batch_size, config.INPUT_DIM).to(config.DEVICE)
    
    # Forward pass
    with torch.no_grad():
        output = model(sample_input)
        print(f"✓ Input shape: {sample_input.shape}")
        print(f"✓ Output shape: {output.shape}")
        print(f"✓ Parameters: {model.count_parameters():,}")
    
    print("\n🎉 STEP 3 COMPLETED SUCCESSFULLY!")
    print("✅ Hybrid 1D-CNN + LSTM model is working!")
    
except Exception as e:
    print(f"❌ STEP 3 FAILED: {e}")
    import traceback
    traceback.print_exc()