import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import os

class BaseIDSModel(nn.Module):
    def __init__(self, config):
        super(BaseIDSModel, self).__init__()
        self.config = config
        self.device = config.DEVICE
        
    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward method")
        
    def save_model(self, path):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, path)
        print(f"Model saved to {path}")
        
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")
        
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def train_epoch(self, train_loader, criterion, optimizer):
        self.train()
        total_loss = 0
        predictions, true_labels = [], []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = self(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            predictions.extend(pred.cpu().numpy())
            true_labels.extend(target.cpu().numpy())
            
        accuracy = accuracy_score(true_labels, predictions)
        return total_loss / len(train_loader), accuracy
    
    def evaluate(self, test_loader, criterion):
        self.eval()
        total_loss = 0
        predictions, true_labels = [], []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self(data)
                loss = criterion(output, target)
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                predictions.extend(pred.cpu().numpy())
                true_labels.extend(target.cpu().numpy())
        
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='binary', zero_division=0)
        recall = recall_score(true_labels, predictions, average='binary', zero_division=0)
        f1 = f1_score(true_labels, predictions, average='binary', zero_division=0)
        
        return {
            'loss': total_loss / len(test_loader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def quick_test(self, test_loader):
        """Quick test to verify model is working"""
        self.eval()
        test_batch = next(iter(test_loader))
        data, target = test_batch
        data, target = data.to(self.device), target.to(self.device)
        
        with torch.no_grad():
            output = self(data)
            pred = output.argmax(dim=1)
            accuracy = (pred == target).float().mean()
        
        print(f"Quick test - Batch accuracy: {accuracy.item():.4f}")
        return accuracy.item()