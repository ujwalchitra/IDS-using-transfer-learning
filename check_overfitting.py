# fast_overfitting_check.py - Quick overfitting diagnosis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import time

print("⚡ FAST OVERFITTING DIAGNOSIS")
print("="*40)

def quick_check(file_path, dataset_name, sample_size=10000):
    print(f"\n📊 Quick check: {dataset_name}")
    start_time = time.time()
    
    # Load only first few rows to check structure
    df_sample = pd.read_csv(file_path, nrows=5)
    print(f"Columns: {len(df_sample.columns)}")
    print(f"Sample columns: {list(df_sample.columns)[:10]}...")
    
    # Check for obvious target leakage
    suspicious_cols = [col for col in df_sample.columns 
                      if any(x in col.lower() for x in ['label', 'class', 'target', 'attack', 'encoded'])]
    print(f"Suspicious columns: {suspicious_cols}")
    
    # Load small sample for analysis
    df = pd.read_csv(file_path, nrows=sample_size)
    
    # Find target column
    target_col = None
    for col in ['Label', 'label', 'Attack_Class', 'attack']:
        if col in df.columns:
            target_col = col
            break
    if not target_col:
        target_col = df.columns[-1]
    
    print(f"Target column: {target_col}")
    
    # Prepare features and target
    X = df.select_dtypes(include=[np.number])
    y = df[target_col]
    
    # Convert to binary if needed
    if y.dtype == 'object':
        y = y.apply(lambda x: 0 if 'benign' in str(x).lower() else 1)
    
    print(f"Sample size: {len(X)}, Features: {len(X.columns)}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # QUICK MODEL TEST
    X_train, X_test, y_train, y_test = train_test_split(
        X.fillna(0), y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Ultra-fast Random Forest
    rf = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    train_score = rf.score(X_train, y_train)
    test_score = rf.score(X_test, y_test)
    
    print(f"🚀 Quick Model Results:")
    print(f"   Train Accuracy: {train_score:.4f}")
    print(f"   Test Accuracy:  {test_score:.4f}")
    print(f"   Overfit Gap:    {train_score - test_score:.4f}")
    
    if train_score > 0.95:
        print("   ⚠️  HIGH ACCURACY - Possible overfitting!")
    
    # Check for duplicate samples (quick check)
    duplicates = df.duplicated().sum()
    print(f"   Duplicate rows: {duplicates} ({duplicates/len(df)*100:.1f}%)")
    
    elapsed = time.time() - start_time
    print(f"   Analysis time: {elapsed:.1f}s")
    
    return train_score, test_score

# MAIN EXECUTION
print("🔍 Running quick diagnostics...")

try:
    # Check pretrain data
    train1, test1 = quick_check(
        r"D:\IoT-DDoS-Detection\02_processed_data\cleaned_intrusion_dataset.csv",
        "PRETRAIN DATA",
        sample_size=5000  # Even smaller sample
    )
    
    # Check finetune data  
    train2, test2 = quick_check(
        r"D:\IoT-DDoS-Detection\02_processed_data\ut2024_finetune_sample.csv", 
        "FINETUNE DATA",
        sample_size=5000
    )
    
    print(f"\n🎯 SUMMARY:")
    print(f"Pretrain - Train: {train1:.4f}, Test: {test1:.4f}, Gap: {train1-test1:.4f}")
    print(f"Finetune - Train: {train2:.4f}, Test: {test2:.4f}, Gap: {train2-test2:.4f}")
    
    if train1 > 0.95 or train2 > 0.95:
        print(f"\n🚨 LIKELY OVERFITTING REASONS:")
        print("1. Target leakage in features (columns with 'label', 'class', 'attack' in name)")
        print("2. Dataset too easy to separate")
        print("3. Duplicate samples")
        print("4. Time-based data leakage")
        
except Exception as e:
    print(f"❌ Error: {e}")
    print("Trying alternative approach...")
    
    # Alternative: Just check file structure
    try:
        df = pd.read_csv(r"D:\IoT-DDoS-Detection\02_processed_data\cleaned_intrusion_dataset.csv", nrows=3)
        print(f"File loaded. Columns: {list(df.columns)}")
        print(f"Check for leakage in: {[col for col in df.columns if 'label' in col.lower() or 'class' in col.lower()]}")
    except:
        print("Could not load file")