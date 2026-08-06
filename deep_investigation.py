# deep_investigation.py - Find why data is still too easy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("🔍 DEEP DATA INVESTIGATION")
print("="*50)

def deep_analyze(file_path, name):
    print(f"\n📊 DEEP ANALYSIS: {name}")
    
    # Load more data
    df = pd.read_csv(file_path, nrows=10000)
    
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Check target distribution
    target_col = 'Label'
    if target_col in df.columns:
        print(f"\n🎯 TARGET ANALYSIS:")
        print(f"Target value counts:")
        print(df[target_col].value_counts())
        
        # Check if binary conversion worked
        if df[target_col].dtype == 'object':
            unique_vals = df[target_col].unique()
            print(f"Unique target values: {unique_vals}")
            
            # Check if it's already mostly one class
            benign_count = df[target_col].str.contains('benign', case=False, na=False).sum()
            attack_count = len(df) - benign_count
            print(f"Benign samples: {benign_count}")
            print(f"Attack samples: {attack_count}")
    
    # Check feature separability
    print(f"\n📈 FEATURE ANALYSIS:")
    
    # Analyze first 10 features
    feature_cols = [col for col in df.columns if col != target_col]
    print(f"Analyzing {len(feature_cols)} features...")
    
    # Check if any single feature can separate classes
    if target_col in df.columns and df[target_col].dtype == 'object':
        for col in feature_cols[:10]:  # Check first 10 features
            if df[col].dtype in ['int64', 'float64']:
                # Calculate separation power
                benign_vals = df[df[target_col].str.contains('benign', case=False, na=False)][col]
                attack_vals = df[~df[target_col].str.contains('benign', case=False, na=False)][col]
                
                if len(benign_vals) > 0 and len(attack_vals) > 0:
                    benign_mean = benign_vals.mean()
                    attack_mean = attack_vals.mean()
                    separation = abs(benign_mean - attack_mean) / (benign_vals.std() + attack_vals.std() + 1e-8)
                    
                    if separation > 5:  # Very high separation
                        print(f"🚨 HIGH SEPARATION: {col} - separation: {separation:.2f}")
                        print(f"   Benign mean: {benign_mean:.2f}, Attack mean: {attack_mean:.2f}")
    
    # Check for constant features
    print(f"\n🔍 CONSTANT/LOW VARIANCE FEATURES:")
    for col in feature_cols[:20]:
        if df[col].dtype in ['int64', 'float64']:
            unique_vals = df[col].nunique()
            if unique_vals <= 3:
                print(f"  {col}: {unique_vals} unique values - {df[col].unique()}")
    
    return df

def check_data_balance(file_path, name):
    print(f"\n⚖️ CHECKING DATA BALANCE: {name}")
    
    # Load larger sample
    df = pd.read_csv(file_path)
    
    target_col = 'Label'
    if target_col in df.columns:
        # Check full distribution
        value_counts = df[target_col].value_counts()
        print(f"Full dataset target distribution:")
        for val, count in value_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {val}: {count:,} samples ({percentage:.1f}%)")
        
        # Check if it's severely imbalanced
        if len(value_counts) == 1:
            print(f"🚨 CRITICAL: Only ONE class in entire dataset!")
            print(f"   All samples are: {value_counts.index[0]}")
        elif len(value_counts) == 2:
            ratio = max(value_counts) / min(value_counts)
            if ratio > 10:
                print(f"🚨 SEVERE IMBALANCE: Ratio {ratio:.1f}:1")

# Run deep analysis
print("🔍 ANALYZING PRETRAIN DATA...")
df_pretrain = deep_analyze(
    r"D:\IoT-DDoS-Detection\02_processed_data\cleaned_intrusion_dataset_FIXED.csv",
    "PRETRAIN"
)
check_data_balance(
    r"D:\IoT-DDoS-Detection\02_processed_data\cleaned_intrusion_dataset_FIXED.csv",
    "PRETRAIN"
)

print("\n🔍 ANALYZING FINETUNE DATA...")
df_finetune = deep_analyze(
    r"D:\IoT-DDoS-Detection\02_processed_data\ut2024_finetune_sample_FIXED.csv",
    "FINETUNE"
)
check_data_balance(
    r"D:\IoT-DDoS-Detection\02_processed_data\ut2024_finetune_sample_FIXED.csv",
    "FINETUNE"
)

print(f"\n🎯 PROBLEM IDENTIFICATION:")
print("1. Check if datasets have only ONE class")
print("2. Check if features are too easily separable") 
print("3. Check if data is synthetic/artificial")
print("4. Check for time-based leakage")