# models/hybrid_1dcnn_bilstm.py
import torch
import torch.nn as nn
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.base_model import BaseIDSModel
except ImportError:
    # Fallback
    class BaseIDSModel(nn.Module):
        def __init__(self, config):
            super(BaseIDSModel, self).__init__()
            self.config = config
            self.device = config.DEVICE
            
        def count_parameters(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

class Hybrid1DCNN_BiLSTM(BaseIDSModel):
    def __init__(self, config):
        super(Hybrid1DCNN_BiLSTM, self).__init__(config)
        
        # Efficient CNN for feature extraction
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.AdaptiveAvgPool1d(25),
            nn.Dropout(config.DROPOUT)
        )
        
        # Bidirectional LSTM for capturing both forward and backward patterns
        self.bilstm = nn.LSTM(
            input_size=64,
            hidden_size=config.HIDDEN_DIM // 2,  # Half because bidirectional
            num_layers=1,
            batch_first=True,
            bidirectional=True,  # This makes it BiLSTM
            dropout=config.DROPOUT
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, 32),  # Hidden dim is doubled due to bidirectional
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(32, config.NUM_CLASSES)
        )
        
        print(f"✓ Hybrid1DCNN_BiLSTM initialized")
        print(f"  - Parameters: {self.count_parameters():,}")
        print(f"  - Bidirectional: Yes")
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Reshape for CNN
        x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        
        # CNN feature extraction
        x = self.cnn_layers(x)  # (batch_size, 64, 25)
        
        # Prepare for BiLSTM
        x = x.permute(0, 2, 1)  # (batch_size, 25, 64)
        
        # Bidirectional LSTM processing
        lstm_out, (hidden, _) = self.bilstm(x)
        
        # Combine bidirectional outputs: hidden shape is (2, batch_size, hidden_dim//2)
        # We concatenate the forward and backward final hidden states
        hidden_forward = hidden[0]  # (batch_size, hidden_dim//2)
        hidden_backward = hidden[1]  # (batch_size, hidden_dim//2)
        x = torch.cat((hidden_forward, hidden_backward), dim=1)  # (batch_size, hidden_dim)
        
        # Classification
        x = self.classifier(x)
        
        return x