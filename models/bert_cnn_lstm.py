# models/bert_cnn_lstm.py
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

class BERT_CNN_LSTM(BaseIDSModel):
    def __init__(self, config):
        super(BERT_CNN_LSTM, self).__init__(config)
        
        # BERT-like feature embedding (lightweight)
        self.embedding = nn.Sequential(
            nn.Linear(config.INPUT_DIM, config.EMBEDDING_DIM),
            nn.LayerNorm(config.EMBEDDING_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT)
        )
        
        # Compact CNN for feature extraction
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.AdaptiveAvgPool1d(20),
            nn.Dropout(config.DROPOUT)
        )
        
        # LSTM for temporal patterns
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=config.HIDDEN_DIM,
            num_layers=1,
            batch_first=True,
            dropout=config.DROPOUT
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, 32),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(32, config.NUM_CLASSES)
        )
        
        print(f"✓ BERT_CNN_LSTM initialized")
        print(f"  - Parameters: {self.count_parameters():,}")
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # BERT-like embedding
        x = self.embedding(x)  # (batch_size, embedding_dim)
        
        # Reshape for CNN
        x = x.unsqueeze(1)  # (batch_size, 1, embedding_dim)
        
        # CNN feature extraction
        x = self.cnn_layers(x)  # (batch_size, 64, 20)
        
        # Prepare for LSTM
        x = x.permute(0, 2, 1)  # (batch_size, 20, 64)
        
        # LSTM processing
        lstm_out, (hidden, _) = self.lstm(x)
        
        # Use last hidden state
        x = hidden[-1]  # (batch_size, hidden_dim)
        
        # Classification
        x = self.classifier(x)
        
        return x