# train_transfer_learning.py
import torch
import torch.nn as nn
from torch.optim import Adam
import time
import json
import os
import pandas as pd
from datetime import datetime

# Import all models
from models.hybrid_1dcnn_lstm import Hybrid1DCNN_LSTM
from models.bert_cnn_lstm import BERT_CNN_LSTM
from models.hybrid_1dcnn_bilstm import Hybrid1DCNN_BiLSTM
from utils.data_loader import IDSDataLoader
from config import Config

class TransferLearningTrainer:
    def __init__(self, config):
        self.config = config
        self.data_loader = IDSDataLoader(config)
        self.results = {}
        self.training_history = {}
        
        # Create directories
        os.makedirs('results', exist_ok=True)
        os.makedirs('saved_models', exist_ok=True)
        
    def train_transfer_learning(self, model_class, model_name):
        """Complete transfer learning: Pretrain → Fine-tune → Evaluate"""
        print(f"\n{'='*60}")
        print(f"🚀 TRANSFER LEARNING: {model_name}")
        print(f"{'='*60}")
        
        total_start = time.time()
        model_results = {
            'model_name': model_name,
            'parameters': 0,
            'pretrain_metrics': {},
            'finetune_metrics': {},
            'transfer_learning_gain': {},
            'training_time': 0
        }
        
        # Initialize model
        model = model_class(self.config).to(self.config.DEVICE)
        total_parameters = model.count_parameters()
        model_results['parameters'] = total_parameters
        print(f"📊 Model Parameters: {total_parameters:,}")
        
        criterion = nn.CrossEntropyLoss()
        
        # ==================== PHASE 1: PRETRAINING ====================
        print(f"\n🎯 PHASE 1: Pretraining ({self.config.PRETRAIN_EPOCHS} epochs)")
        print("📥 Loading pretraining data...")
        
        X_pretrain, y_pretrain = self.data_loader.load_and_preprocess_data(
            self.config.PRETRAIN_PATH, "pretrain"
        )
        pretrain_loader, pretrain_val_loader = self.data_loader.create_dataloaders(
            X_pretrain, y_pretrain, batch_size=self.config.BATCH_SIZE
        )
        
        # Pretraining optimizer
        pretrain_optimizer = Adam(model.parameters(), lr=self.config.LEARNING_RATE)
        
        best_pretrain_acc = 0
        pretrain_start = time.time()
        
        for epoch in range(self.config.PRETRAIN_EPOCHS):
            epoch_start = time.time()
            
            # Training
            model.train()
            train_loss, train_acc = 0, 0
            for data, target in pretrain_loader:
                data, target = data.to(self.config.DEVICE), target.to(self.config.DEVICE)
                pretrain_optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                pretrain_optimizer.step()
                
                train_loss += loss.item()
                pred = output.argmax(dim=1)
                train_acc += (pred == target).float().mean().item()
            
            # Validation
            model.eval()
            val_loss, val_acc = 0, 0
            with torch.no_grad():
                for data, target in pretrain_val_loader:
                    data, target = data.to(self.config.DEVICE), target.to(self.config.DEVICE)
                    output = model(data)
                    val_loss += criterion(output, target).item()
                    pred = output.argmax(dim=1)
                    val_acc += (pred == target).float().mean().item()
            
            train_loss /= len(pretrain_loader)
            train_acc /= len(pretrain_loader)
            val_loss /= len(pretrain_val_loader)
            val_acc /= len(pretrain_val_loader)
            
            epoch_time = time.time() - epoch_start
            
            if (epoch + 1) % 5 == 0:
                print(f'  Epoch {epoch+1}/{self.config.PRETRAIN_EPOCHS} | Time: {epoch_time:.1f}s | '
                      f'Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}')
            
            if val_acc > best_pretrain_acc:
                best_pretrain_acc = val_acc
                # Save pretrained model
                torch.save(model.state_dict(), f'saved_models/{model_name}_pretrained.pth')
        
        pretrain_time = time.time() - pretrain_start
        print(f"✅ Pretraining completed: Best Val Acc = {best_pretrain_acc:.4f}, Time = {pretrain_time:.1f}s")
        
        # ==================== PHASE 2: FINE-TUNING ====================
        print(f"\n🎯 PHASE 2: Fine-tuning ({self.config.FINETUNE_EPOCHS} epochs)")
        print("📥 Loading fine-tuning data...")
        
        # Load pretrained weights
        model.load_state_dict(torch.load(f'saved_models/{model_name}_pretrained.pth'))
        
        X_finetune, y_finetune = self.data_loader.load_and_preprocess_data(
            self.config.FINETUNE_PATH, "finetune"
        )
        finetune_loader, finetune_test_loader = self.data_loader.create_dataloaders(
            X_finetune, y_finetune, batch_size=self.config.BATCH_SIZE
        )
        
        # Fine-tuning with smaller learning rate
        finetune_optimizer = Adam(model.parameters(), lr=self.config.LEARNING_RATE/10)
        
        best_finetune_acc = 0
        finetune_start = time.time()
        
        for epoch in range(self.config.FINETUNE_EPOCHS):
            epoch_start = time.time()
            
            # Fine-tuning
            model.train()
            train_loss, train_acc = 0, 0
            for data, target in finetune_loader:
                data, target = data.to(self.config.DEVICE), target.to(self.config.DEVICE)
                finetune_optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                finetune_optimizer.step()
                
                train_loss += loss.item()
                pred = output.argmax(dim=1)
                train_acc += (pred == target).float().mean().item()
            
            # Test evaluation
            model.eval()
            test_loss, test_acc = 0, 0
            with torch.no_grad():
                for data, target in finetune_test_loader:
                    data, target = data.to(self.config.DEVICE), target.to(self.config.DEVICE)
                    output = model(data)
                    test_loss += criterion(output, target).item()
                    pred = output.argmax(dim=1)
                    test_acc += (pred == target).float().mean().item()
            
            train_loss /= len(finetune_loader)
            train_acc /= len(finetune_loader)
            test_loss /= len(finetune_test_loader)
            test_acc /= len(finetune_test_loader)
            
            epoch_time = time.time() - epoch_start
            
            if (epoch + 1) % 3 == 0:
                print(f'  Epoch {epoch+1}/{self.config.FINETUNE_EPOCHS} | Time: {epoch_time:.1f}s | '
                      f'Train Loss: {train_loss:.4f} | Test Acc: {test_acc:.4f}')
            
            if test_acc > best_finetune_acc:
                best_finetune_acc = test_acc
                # Save fine-tuned model
                torch.save(model.state_dict(), f'saved_models/{model_name}_finetuned.pth')
        
        finetune_time = time.time() - finetune_start
        total_time = time.time() - total_start
        
        # ==================== RESULTS ====================
        model_results['pretrain_metrics'] = {
            'best_accuracy': best_pretrain_acc,
            'training_time': pretrain_time
        }
        
        model_results['finetune_metrics'] = {
            'best_accuracy': best_finetune_acc,
            'training_time': finetune_time
        }
        
        model_results['transfer_learning_gain'] = {
            'accuracy_improvement': best_finetune_acc - best_pretrain_acc,
            'total_training_time': total_time
        }
        
        print(f"\n✅ TRANSFER LEARNING COMPLETED: {model_name}")
        print(f"   Pretrain Acc: {best_pretrain_acc:.4f}")
        print(f"   Finetune Acc: {best_finetune_acc:.4f}")
        print(f"   Improvement:  +{best_finetune_acc - best_pretrain_acc:.4f}")
        print(f"   Total Time:   {total_time:.1f}s")
        
        return model_results
    
    def train_all_models(self):
        """Train all models and compare results"""
        models = {
            'Hybrid_1DCNN_LSTM': Hybrid1DCNN_LSTM,
            'BERT_CNN_LSTM': BERT_CNN_LSTM, 
            'Hybrid_1DCNN_BiLSTM': Hybrid1DCNN_BiLSTM
        }
        
        print("🤖 STARTING TRANSFER LEARNING COMPARISON")
        print("Models to train:", list(models.keys()))
        
        for name, model_class in models.items():
            results = self.train_transfer_learning(model_class, name)
            self.results[name] = results
        
        # Save comparison results
        self.save_comparison_results()
        
    def save_comparison_results(self):
        """Save all results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results as JSON
        with open(f'results/transfer_learning_results_{timestamp}.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save comparison table as CSV
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Parameters': results['parameters'],
                'Pretrain_Accuracy': results['pretrain_metrics']['best_accuracy'],
                'Finetune_Accuracy': results['finetune_metrics']['best_accuracy'],
                'Improvement': results['transfer_learning_gain']['accuracy_improvement'],
                'Total_Time_Seconds': results['transfer_learning_gain']['total_training_time'],
                'Pretrain_Time': results['pretrain_metrics']['training_time'],
                'Finetune_Time': results['finetune_metrics']['training_time']
            })
        
        df = pd.DataFrame(comparison_data)
        df.to_csv(f'results/model_comparison_{timestamp}.csv', index=False)
        
        # Print summary
        print(f"\n{'='*80}")
        print("📊 TRANSFER LEARNING COMPARISON RESULTS")
        print(f"{'='*80}")
        print(df.round(4))
        
        # Find best model
        best_model = df.loc[df['Finetune_Accuracy'].idxmax()]
        print(f"\n🏆 BEST MODEL: {best_model['Model']}")
        print(f"   Accuracy: {best_model['Finetune_Accuracy']:.4f}")
        print(f"   Improvement: +{best_model['Improvement']:.4f}")
        
        print(f"\n💾 Results saved to:")
        print(f"   JSON: results/transfer_learning_results_{timestamp}.json")
        print(f"   CSV:  results/model_comparison_{timestamp}.csv")

if __name__ == "__main__":
    config = Config()
    trainer = TransferLearningTrainer(config)
    trainer.train_all_models()
    