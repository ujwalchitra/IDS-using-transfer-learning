# train_quick_test.py - Fixed array handling
import torch
import torch.nn as nn
from torch.optim import Adam
import time
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime

from models.hybrid_1dcnn_lstm import Hybrid1DCNN_LSTM
from models.bert_cnn_lstm import BERT_CNN_LSTM
from models.hybrid_1dcnn_bilstm import Hybrid1DCNN_BiLSTM
from utils.data_loader import IDSDataLoader
from config import Config

class SubstantialTransferLearning:
    def __init__(self, config):
        self.config = config
        self.data_loader = IDSDataLoader(config)
        self.results = {}
        
        os.makedirs('results', exist_ok=True)
        os.makedirs('saved_models', exist_ok=True)
    
    def load_substantial_sample(self, file_path, dataset_type, sample_fraction=0.5):
        """Load substantial sample (half of the data)"""
        print(f"📥 Loading substantial sample ({sample_fraction*100}%) from {file_path}...")
        
        # Load full data - this returns numpy arrays from data_loader
        X_full, y_full = self.data_loader.load_and_preprocess_data(file_path, dataset_type)
        
        # Convert to numpy arrays if they are DataFrames
        if hasattr(X_full, 'values'):
            X_full = X_full.values
        if hasattr(y_full, 'values'):
            y_full = y_full.values
        
        # Calculate sample size
        sample_size = int(len(X_full) * sample_fraction)
        print(f"✓ Sample size: {sample_size:,} out of {len(X_full):,} total samples")
        
        # Take balanced sample
        benign_indices = np.where(y_full == 0)[0]
        attack_indices = np.where(y_full == 1)[0]
        
        # Sample from each class (maintain original distribution)
        benign_sample_size = int(sample_size * len(benign_indices) / len(X_full))
        attack_sample_size = sample_size - benign_sample_size
        
        print(f"✓ Sampling {benign_sample_size:,} benign and {attack_sample_size:,} attack samples")
        
        benign_sample = np.random.choice(benign_indices, benign_sample_size, replace=False)
        attack_sample = np.random.choice(attack_indices, attack_sample_size, replace=False)
        
        # Combine
        sample_indices = np.concatenate([benign_sample, attack_sample])
        np.random.shuffle(sample_indices)
        
        # Use numpy array indexing
        X_sample = X_full[sample_indices]
        y_sample = y_full[sample_indices]
        
        print(f"✓ Final sample shape: {X_sample.shape}")
        print(f"✓ Class balance: Benign: {np.sum(y_sample == 0):,}, Attack: {np.sum(y_sample == 1):,}")
        return X_sample, y_sample
    
    def substantial_train(self, model_class, model_name, pretrain_fraction=0.3, finetune_fraction=0.5):
        """Substantial training with meaningful data"""
        print(f"\n{'='*60}")
        print(f"🚀 SUBSTANTIAL TRAINING: {model_name}")
        print(f"   Using {pretrain_fraction*100}% pretrain data, {finetune_fraction*100}% finetune data")
        print(f"{'='*60}")
        
        start_time = time.time()
        model = model_class(self.config).to(self.config.DEVICE)
        criterion = nn.CrossEntropyLoss()
        
        print(f"📊 Model Parameters: {model.count_parameters():,}")
        
        # PHASE 1: Substantial Pretraining
        print(f"\n🎯 PHASE 1: Pretraining ({self.config.PRETRAIN_EPOCHS} epochs)")
        X_pretrain, y_pretrain = self.load_substantial_sample(
            self.config.PRETRAIN_PATH, "pretrain", pretrain_fraction
        )
        
        train_loader, val_loader = self.data_loader.create_dataloaders(
            X_pretrain, y_pretrain, batch_size=128, test_size=0.2
        )
        
        optimizer = Adam(model.parameters(), lr=self.config.LEARNING_RATE)
        
        # Training with more epochs
        best_pretrain_acc = 0
        pretrain_accuracies = []
        
        for epoch in range(self.config.PRETRAIN_EPOCHS):
            epoch_start = time.time()
            
            # Training
            model.train()
            train_loss, train_correct, total_samples = 0, 0, 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.config.DEVICE), target.to(self.config.DEVICE)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pred = output.argmax(dim=1)
                train_correct += (pred == target).sum().item()
                total_samples += len(data)
                
                # Print batch progress for large datasets
                if batch_idx % 50 == 0:
                    print(f"    Batch {batch_idx}/{len(train_loader)}", end='\r')
            
            # Validation
            model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.config.DEVICE), target.to(self.config.DEVICE)
                    output = model(data)
                    pred = output.argmax(dim=1)
                    val_correct += (pred == target).sum().item()
                    val_total += len(data)
            
            train_acc = train_correct / total_samples
            val_acc = val_correct / val_total
            epoch_time = time.time() - epoch_start
            
            pretrain_accuracies.append(val_acc)
            
            print(f"  Epoch {epoch+1}/{self.config.PRETRAIN_EPOCHS} | Time: {epoch_time:.1f}s | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            
            if val_acc > best_pretrain_acc:
                best_pretrain_acc = val_acc
                torch.save(model.state_dict(), f'saved_models/{model_name}_pretrained.pth')
        
        # PHASE 2: Substantial Fine-tuning
        print(f"\n🎯 PHASE 2: Fine-tuning ({self.config.FINETUNE_EPOCHS} epochs)")
        
        # Load best pretrained weights
        model.load_state_dict(torch.load(f'saved_models/{model_name}_pretrained.pth', map_location=self.config.DEVICE))
        
        X_finetune, y_finetune = self.load_substantial_sample(
            self.config.FINETUNE_PATH, "finetune", finetune_fraction
        )
        
        finetune_loader, test_loader = self.data_loader.create_dataloaders(
            X_finetune, y_finetune, batch_size=128, test_size=0.2
        )
        
        # Fine-tune with smaller LR
        finetune_optimizer = Adam(model.parameters(), lr=self.config.LEARNING_RATE/10)
        
        # Fine-tuning
        best_finetune_acc = 0
        finetune_accuracies = []
        
        for epoch in range(self.config.FINETUNE_EPOCHS):
            epoch_start = time.time()
            
            # Fine-tuning
            model.train()
            finetune_correct, finetune_total = 0, 0
            for batch_idx, (data, target) in enumerate(finetune_loader):
                data, target = data.to(self.config.DEVICE), target.to(self.config.DEVICE)
                finetune_optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                finetune_optimizer.step()
                
                pred = output.argmax(dim=1)
                finetune_correct += (pred == target).sum().item()
                finetune_total += len(data)
                
                if batch_idx % 20 == 0:
                    print(f"    Batch {batch_idx}/{len(finetune_loader)}", end='\r')
            
            # Test evaluation
            model.eval()
            test_correct, test_total = 0, 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.config.DEVICE), target.to(self.config.DEVICE)
                    output = model(data)
                    pred = output.argmax(dim=1)
                    test_correct += (pred == target).sum().item()
                    test_total += len(data)
            
            finetune_acc = finetune_correct / finetune_total
            test_acc = test_correct / test_total
            epoch_time = time.time() - epoch_start
            
            finetune_accuracies.append(test_acc)
            
            print(f"  Epoch {epoch+1}/{self.config.FINETUNE_EPOCHS} | Time: {epoch_time:.1f}s | "
                  f"Finetune Acc: {finetune_acc:.4f} | Test Acc: {test_acc:.4f}")
            
            if test_acc > best_finetune_acc:
                best_finetune_acc = test_acc
                torch.save(model.state_dict(), f'saved_models/{model_name}_finetuned.pth')
        
        total_time = time.time() - start_time
        
        results = {
            'model_name': model_name,
            'parameters': model.count_parameters(),
            'pretrain_samples': len(X_pretrain),
            'finetune_samples': len(X_finetune),
            'best_pretrain_acc': best_pretrain_acc,
            'best_finetune_acc': best_finetune_acc,
            'improvement': best_finetune_acc - best_pretrain_acc,
            'total_time': total_time
        }
        
        print(f"\n✅ {model_name} TRANSFER LEARNING COMPLETED:")
        print(f"   Pretrain Samples: {results['pretrain_samples']:,}")
        print(f"   Finetune Samples: {results['finetune_samples']:,}")
        print(f"   Pretrain Acc:     {results['best_pretrain_acc']:.4f}")
        print(f"   Finetune Acc:     {results['best_finetune_acc']:.4f}")
        print(f"   Improvement:      +{results['improvement']:.4f}")
        print(f"   Total Time:       {total_time/60:.1f} minutes")
        
        return results
    
    def run_substantial_comparison(self):
        """Compare all models with substantial data"""
        models = {
            '1DCNN_LSTM': Hybrid1DCNN_LSTM,
            'BERT_CNN_LSTM': BERT_CNN_LSTM,
            '1DCNN_BiLSTM': Hybrid1DCNN_BiLSTM
        }
        
        print("🤖 SUBSTANTIAL TRANSFER LEARNING COMPARISON")
        print("Using 30% of pretrain data and 50% of finetune data for optimal results")
        print("=" * 70)
        
        results = {}
        for name, model_class in models.items():
            results[name] = self.substantial_train(model_class, name, 
                                                 pretrain_fraction=0.3,  # 30% of large dataset
                                                 finetune_fraction=0.5)  # 50% of finetune dataset
        
        # Print comprehensive comparison table
        print(f"\n{'='*90}")
        print("📊 TRANSFER LEARNING COMPARISON RESULTS")
        print(f"{'='*90}")
        
        print(f"{'Model':<15} {'Params':<12} {'Pretrain N':<12} {'Finetune N':<12} "
              f"{'Pretrain Acc':<12} {'Finetune Acc':<12} {'Improvement':<12} {'Time (min)':<10}")
        print("-" * 90)
        
        for name, result in results.items():
            print(f"{name:<15} {result['parameters']:<12,} "
                  f"{result['pretrain_samples']:<12,} "
                  f"{result['finetune_samples']:<12,} "
                  f"{result['best_pretrain_acc']:.4f}      "
                  f"{result['best_finetune_acc']:.4f}      "
                  f"+{result['improvement']:.4f}      "
                  f"{result['total_time']/60:.1f}")
        
        # Find best model
        best_model = max(results.items(), key=lambda x: x[1]['best_finetune_acc'])
        print(f"\n🏆 BEST MODEL: {best_model[0]}")
        print(f"   Final Accuracy: {best_model[1]['best_finetune_acc']:.4f}")
        print(f"   Improvement from pretraining: +{best_model[1]['improvement']:.4f}")
        print(f"   Total Parameters: {best_model[1]['parameters']:,}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f'results/transfer_learning_results_{timestamp}.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: results/transfer_learning_results_{timestamp}.json")
        
        return results

if __name__ == "__main__":
    config = Config()
    # Set epochs for substantial training
    config.PRETRAIN_EPOCHS = 8
    config.FINETUNE_EPOCHS = 6
    
    print(f"⚙️  Training Configuration:")
    print(f"   Device: {config.DEVICE}")
    print(f"   Pretrain Epochs: {config.PRETRAIN_EPOCHS}")
    print(f"   Finetune Epochs: {config.FINETUNE_EPOCHS}")
    print(f"   Batch Size: {config.BATCH_SIZE}")
    
    substantial_trainer = SubstantialTransferLearning(config)
    results = substantial_trainer.run_substantial_comparison()