# fix_data_leakage.py - Remove target leakage from datasets
import pandas as pd
import numpy as np

print("🔧 FIXING DATA LEAKAGE ISSUES")
print("="*40)

def fix_pretrain_data():
    print("\n📁 FIXING PRETRAIN DATA...")
    
    # Load pretrain data
    df = pd.read_csv(r"D:\IoT-DDoS-Detection\02_processed_data\cleaned_intrusion_dataset.csv")
    
    print(f"Original shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")
    
    # REMOVE LEAKAGE COLUMNS
    leakage_cols = ['Attack_Class_Encoded', 'Attack_Class']
    df_fixed = df.drop(columns=leakage_cols, errors='ignore')
    
    print(f"After removing leakage: {df_fixed.shape}")
    print(f"Remaining columns: {len(df_fixed.columns)}")
    
    # Check class distribution
    if 'Label' in df_fixed.columns:
        print(f"Class distribution in 'Label':")
        print(df_fixed['Label'].value_counts())
    
    # Save fixed data
    df_fixed.to_csv(r"D:\IoT-DDoS-Detection\02_processed_data\cleaned_intrusion_dataset_FIXED.csv", index=False)
    print("✅ Fixed pretrain data saved!")
    
    return df_fixed

def fix_finetune_data():
    print("\n📁 FIXING FINETUNE DATA...")
    
    # Load finetune data
    df = pd.read_csv(r"D:\IoT-DDoS-Detection\02_processed_data\ut2024_finetune_sample.csv")
    
    print(f"Original shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")
    
    # Remove timestamp and handle infinity
    df_fixed = df.drop(columns=['Timestamp'], errors='ignore')
    
    # Replace infinity with large numbers
    df_fixed = df_fixed.replace([np.inf, -np.inf], np.nan)
    df_fixed = df_fixed.fillna(0)
    
    print(f"After cleaning: {df_fixed.shape}")
    
    # Check class distribution
    if 'Label' in df_fixed.columns:
        print(f"Class distribution in 'Label':")
        print(df_fixed['Label'].value_counts())
    
    # Save fixed data
    df_fixed.to_csv(r"D:\IoT-DDoS-Detection\02_processed_data\ut2024_finetune_sample_FIXED.csv", index=False)
    print("✅ Fixed finetune data saved!")
    
    return df_fixed

def test_fixed_data():
    print("\n🧪 TESTING FIXED DATA...")
    
    # Test pretrain fixed
    df_pretrain = pd.read_csv(r"D:\IoT-DDoS-Detection\02_processed_data\cleaned_intrusion_dataset_FIXED.csv", nrows=1000)
    print(f"Pretrain fixed - Shape: {df_pretrain.shape}")
    if 'Label' in df_pretrain.columns:
        print(f"Classes: {df_pretrain['Label'].value_counts()}")
    
    # Test finetune fixed  
    df_finetune = pd.read_csv(r"D:\IoT-DDoS-Detection\02_processed_data\ut2024_finetune_sample_FIXED.csv", nrows=1000)
    print(f"Finetune fixed - Shape: {df_finetune.shape}")
    if 'Label' in df_finetune.columns:
        print(f"Classes: {df_finetune['Label'].value_counts()}")

# RUN THE FIX
if __name__ == "__main__":
    fix_pretrain_data()
    fix_finetune_data()
    test_fixed_data()
    print("\n🎯 NEXT: Update your config.py to use the FIXED files!")