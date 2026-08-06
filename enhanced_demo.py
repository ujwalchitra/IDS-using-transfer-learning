# fix_sorted_data.py - Fix sorted data issue
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

print("🔧 FIXING SORTED DATA ISSUE")
print("="*40)

def fix_pretrain_data():
    print("🔄 FIXING PRETRAIN DATA (Sorted by class)...")
    
    # Load FULL dataset
    df = pd.read_csv(r"D:\IoT-DDoS-Detection\02_processed_data\cleaned_intrusion_dataset_FIXED.csv")
    
    print(f"Original dataset shape: {df.shape}")
    print("Original class distribution:")
    print(df['Label'].value_counts())
    
    # SHUFFLE the entire dataset
    df_shuffled = shuffle(df, random_state=42)
    
    print(f"\nAfter shuffling - first 10 labels:")
    print(df_shuffled['Label'].head(10).value_counts())
    
    # Save shuffled data
    df_shuffled.to_csv(r"D:\IoT-DDoS-Detection\02_processed_data\cleaned_intrusion_dataset_SHUFFLED.csv", index=False)
    print("✅ Shuffled pretrain data saved!")
    
    return df_shuffled

def fix_finetune_data():
    print("\n🔄 FIXING FINETUNE DATA...")
    
    # Load FULL dataset
    df = pd.read_csv(r"D:\IoT-DDoS-Detection\02_processed_data\ut2024_finetune_sample_FIXED.csv")
    
    print(f"Original dataset shape: {df.shape}")
    print("Original class distribution:")
    print(df['Label'].value_counts())
    
    # Convert to binary and shuffle
    df_binary = df.copy()
    df_binary['Label'] = df_binary['Label'].apply(lambda x: 0 if 'benign' in str(x).lower() else 1)
    
    df_shuffled = shuffle(df_binary, random_state=42)
    
    print(f"\nAfter shuffling - binary distribution:")
    print(df_shuffled['Label'].value_counts())
    
    # Save shuffled data
    df_shuffled.to_csv(r"D:\IoT-DDoS-Detection\02_processed_data\ut2024_finetune_sample_SHUFFLED.csv", index=False)
    print("✅ Shuffled finetune data saved!")
    
    return df_shuffled

def test_shuffled_data():
    print("\n🧪 TESTING SHUFFLED DATA...")
    
    # Test pretrain shuffled
    df_pretrain = pd.read_csv(r"D:\IoT-DDoS-Detection\02_processed_data\cleaned_intrusion_dataset_SHUFFLED.csv", nrows=5000)
    print(f"Pretrain shuffled - Shape: {df_pretrain.shape}")
    print(f"First 5000 samples distribution:")
    print(df_pretrain['Label'].value_counts())
    
    # Test finetune shuffled
    df_finetune = pd.read_csv(r"D:\IoT-DDoS-Detection\02_processed_data\ut2024_finetune_sample_SHUFFLED.csv", nrows=5000)
    print(f"\nFinetune shuffled - Shape: {df_finetune.shape}")
    print(f"First 5000 samples distribution:")
    print(df_finetune['Label'].value_counts())

# Run the fix
pretrain_fixed = fix_pretrain_data()
finetune_fixed = fix_finetune_data()
test_shuffled_data()

print(f"\n🎯 NEXT: Update config.py to use SHUFFLED files!")