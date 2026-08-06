# check_dataset_balance.py - Check current dataset balance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("🔍 DATASET BALANCE ANALYSIS")
print("="*50)

def analyze_balance(file_path, dataset_name, sample_size=50000):
    print(f"\n📊 Analyzing: {dataset_name}")
    print("-" * 40)
    
    try:
        # Load data
        df = pd.read_csv(file_path, nrows=sample_size)
        
        print(f"Loaded: {len(df)} samples")
        
        # Check target column
        target_col = 'Label'
        if target_col not in df.columns:
            print(f"❌ Target column '{target_col}' not found!")
            return None
        
        # Analyze distribution
        value_counts = df[target_col].value_counts()
        total_samples = len(df)
        
        print(f"Class Distribution:")
        print("-" * 30)
        
        # For binary classification analysis
        if value_counts.nunique() <= 10:  # If few unique values, might be binary
            # Try to detect binary pattern
            benign_keywords = ['benign', 'normal', 'legitimate']
            attack_keywords = ['ddos', 'dos', 'bot', 'brute', 'sql', 'xss', 'portscan', 'attack']
            
            benign_count = 0
            attack_count = 0
            other_count = 0
            
            for label, count in value_counts.items():
                label_str = str(label).lower()
                if any(keyword in label_str for keyword in benign_keywords):
                    benign_count += count
                    print(f"  BENIGN ({label}): {count:,} samples ({count/total_samples*100:.1f}%)")
                elif any(keyword in label_str for keyword in attack_keywords):
                    attack_count += count
                    print(f"  ATTACK ({label}): {count:,} samples ({count/total_samples*100:.1f}%)")
                else:
                    other_count += count
                    print(f"  OTHER  ({label}): {count:,} samples ({count/total_samples*100:.1f}%)")
            
            # Binary balance analysis
            if benign_count + attack_count > 0:
                total_binary = benign_count + attack_count
                benign_percent = benign_count / total_binary * 100
                attack_percent = attack_count / total_binary * 100
                
                print(f"\n🎯 BINARY CLASSIFICATION ANALYSIS:")
                print(f"  Benign: {benign_count:,} samples ({benign_percent:.1f}%)")
                print(f"  Attack: {attack_count:,} samples ({attack_percent:.1f}%)")
                print(f"  Balance Ratio: {max(benign_count, attack_count)/min(benign_count, attack_count):.2f}:1")
                
                if 40 <= benign_percent <= 60 and 40 <= attack_percent <= 60:
                    print("  ✅ WELL BALANCED (40-60% each)")
                elif 30 <= benign_percent <= 70 and 30 <= attack_percent <= 70:
                    print("  ⚠️  MODERATELY BALANCED (30-70% each)")
                else:
                    print("  🚨 IMBALANCED (<30% or >70% for one class)")
            
            if other_count > 0:
                print(f"  Note: {other_count:,} samples couldn't be classified as Benign/Attack")
        
        else:
            # Multi-class analysis
            print("Multi-class distribution:")
            for label, count in value_counts.items():
                print(f"  {label}: {count:,} samples ({count/total_samples*100:.1f}%)")
        
        return value_counts
        
    except Exception as e:
        print(f"❌ Error analyzing {dataset_name}: {e}")
        return None

def check_full_dataset_balance(file_path, dataset_name):
    print(f"\n📈 Checking FULL dataset balance: {dataset_name}")
    print("-" * 40)
    
    try:
        # Get total count without loading entire file
        chunk_size = 50000
        total_counts = {}
        total_rows = 0
        
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            if 'Label' in chunk.columns:
                chunk_counts = chunk['Label'].value_counts().to_dict()
                for label, count in chunk_counts.items():
                    total_counts[label] = total_counts.get(label, 0) + count
                total_rows += len(chunk)
        
        print(f"Total samples: {total_rows:,}")
        print("Full distribution:")
        for label, count in sorted(total_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = count / total_rows * 100
            print(f"  {label}: {count:,} samples ({percentage:.1f}%)")
            
            # Classify as benign/attack
            label_str = str(label).lower()
            if any(keyword in label_str for keyword in ['benign', 'normal']):
                print(f"    → BENIGN class")
            elif any(keyword in label_str for keyword in ['ddos', 'dos', 'bot', 'attack']):
                print(f"    → ATTACK class")
        
        return total_counts
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def create_balance_visualization(pretrain_counts, finetune_counts):
    print(f"\n📊 CREATING BALANCE VISUALIZATION...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Pretrain data
    if pretrain_counts:
        labels = [str(label) for label in pretrain_counts.keys()]
        sizes = list(pretrain_counts.values())
        
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Pretrain Dataset - Class Distribution')
    
    # Finetune data
    if finetune_counts:
        labels = [str(label) for label in finetune_counts.keys()]
        sizes = list(finetune_counts.values())
        
        ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Finetune Dataset - Class Distribution')
    
    plt.tight_layout()
    plt.savefig('dataset_balance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# Main analysis
print("🔍 ANALYZING CURRENT DATASET BALANCE")
print("="*50)

# Analyze current shuffled datasets
print("🔄 Checking SHUFFLED datasets (current state):")

pretrain_sample = analyze_balance(
    r"D:\IoT-DDoS-Detection\02_processed_data\cleaned_intrusion_dataset_SHUFFLED.csv",
    "PRETRAIN DATA (Shuffled)"
)

finetune_sample = analyze_balance(
    r"D:\IoT-DDoS-Detection\02_processed_data\ut2024_finetune_sample_SHUFFLED.csv",
    "FINETUNE DATA (Shuffled)"
)

print(f"\n{'='*60}")
print("📊 FULL DATASET ANALYSIS")
print(f"{'='*60}")

pretrain_full = check_full_dataset_balance(
    r"D:\IoT-DDoS-Detection\02_processed_data\cleaned_intrusion_dataset_SHUFFLED.csv",
    "PRETRAIN FULL"
)

finetune_full = check_full_dataset_balance(
    r"D:\IoT-DDoS-Detection\02_processed_data\ut2024_finetune_sample_SHUFFLED.csv",
    "FINETUNE FULL"
)

# Create visualization
create_balance_visualization(pretrain_sample, finetune_sample)

print(f"\n🎯 FINAL BALANCE ASSESSMENT:")
print("Based on the shuffled datasets, we can now determine if your data is properly balanced for training!")4