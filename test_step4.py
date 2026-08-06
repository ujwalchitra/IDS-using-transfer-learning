# test_step4.py
import torch
import torch.nn as nn
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("🧪 TESTING STEP 4: BERT + CNN + LSTM Model")

try:
    from config import Config
    from models.bert_cnn_lstm import BERT_CNN_LSTM
    
    config = Config()
    print("✓ Config loaded")
    
    # Create model
    model = BERT_CNN_LSTM(config).to(config.DEVICE)
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
    
    print("\n🎉 STEP 4 COMPLETED SUCCESSFULLY!")
    print("✅ BERT + CNN + LSTM model is working!")
    print("➡️ Ready for Step 5: Create Hybrid 1D-CNN + BiLSTM model")
    
except Exception as e:
    print(f"❌ STEP 4 FAILED: {e}")
    import traceback
    traceback.print_exc()