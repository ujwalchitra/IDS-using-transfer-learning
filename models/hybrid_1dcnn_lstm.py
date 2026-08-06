import torch
import torch.nn as nn
from .base_model import BaseIDSModel

class Hybrid1DCNN_LSTM(BaseIDSModel):
    def __init__(self, config):
        super(Hybrid1DCNN_LSTM, self).__init__(config)
        
        # Lightweight CNN for feature extraction
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),
            nn.Dropout(config.DROPOUT),
            
            # Second conv block
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.AdaptiveAvgPool1d(25),  # Fixed size output
            nn.Dropout(config.DROPOUT)
        )
        
        # Calculate CNN output size
        self.cnn_output_size = self._get_conv_output(config.INPUT_DIM)
        
        # LSTM for sequence learning
        self.lstm = nn.LSTM(
            input_size=64,  # CNN output channels
            hidden_size=config.HIDDEN_DIM,
            num_layers=1,
            batch_first=True,
            dropout=config.DROPOUT,
            bidirectional=False
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, 32),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(32, config.NUM_CLASSES)
        )
        
        print(f"✓ Hybrid1DCNN_LSTM initialized")
        print(f"  - CNN output size: {self.cnn_output_size}")
        print(f"  - Total parameters: {self.count_parameters():,}")
        
    def _get_conv_output(self, input_dim):
        """Calculate CNN output size"""
        with torch.no_grad():
            x = torch.zeros(1, 1, input_dim)
            x = self.conv_layers(x)
            return x.numel()
    
    def forward(self, x):
        # x shape: (batch_size, input_dim)
        batch_size = x.shape[0]
        
        # Reshape for CNN: (batch_size, 1, input_dim)
        x = x.unsqueeze(1)
        
        # CNN feature extraction
        x = self.conv_layers(x)  # Output: (batch_size, 64, 25)
        
        # Prepare for LSTM: (batch_size, sequence_length, features)
        x = x.permute(0, 2, 1)  # (batch_size, 25, 64)
        
        # LSTM processing
        lstm_out, (hidden, _) = self.lstm(x)
        
        # Use last hidden state
        x = hidden[-1]  # (batch_size, hidden_dim)
        
        # Classification
        x = self.classifier(x)
        
        return x

    def get_model_info(self):
        """Get model information"""
        return {
            'name': 'Hybrid1DCNN_LSTM',
            'parameters': self.count_parameters(),
            'architecture': {
                'cnn_filters': [32, 64],
                'lstm_hidden': self.config.HIDDEN_DIM,
                'input_dim': self.config.INPUT_DIM,
                'output_classes': self.config.NUM_CLASSES
            }
        }