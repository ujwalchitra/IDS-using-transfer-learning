# test_fixed_data.py - Verify the fixed data works
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

print("🧪 VERIFYING FIXED DATA")
print("="*40)

def test_dataset(file_path, name):
    print(f"\n📊 Testing: {name}")
    
    # Load data
    df = pd.read_csv(file_path, nrows=5000)
    
    # Check for target leakage
    leakage_cols = [col for col in df.columns if 'encoded' in col.lower() or 'class' in col.lower()]
    print(f"Remaining suspicious columns: {leakage_cols}")
    
    # Prepare features (exclude target)
    target_col = 'Label'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Convert to binary if needed
    if y.dtype == 'object':
        y = y.apply(lambda x: 0 if 'benign' in str(x).lower() else 1)
    
    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Quick model test
    X_train, X_test, y_train, y_test = train_test_split(
        X.fillna(0), y, test_size=0.3, random_state=42, stratify=y
    )
    
    rf = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    
    train_score = rf.score(X_train, y_train)
    test_score = rf.score(X_test, y_test)
    
    print(f"Model Results:")
    print(f"  Train Accuracy: {train_score:.4f}")
    print(f"  Test Accuracy:  {test_score:.4f}")
    print(f"  Overfit Gap:    {train_score - test_score:.4f}")
    
    if 0.85 <= test_score <= 0.95:
        print("  ✅ REALISTIC ACCURACY - Data is fixed!")
    elif test_score > 0.95:
        print("  ⚠️  Still too high - check for other leakage")
    else:
        print("  📉 Lower accuracy - but more realistic")
    
    return train_score, test_score

# Test both datasets
try:
    train1, test1 = test_dataset(
        r"D:\IoT-DDoS-Detection\02_processed_data\cleaned_intrusion_dataset_FIXED.csv",
        "FIXED PRETRAIN DATA"
    )
    
    train2, test2 = test_dataset(
        r"D:\IoT-DDoS-Detection\02_processed_data\ut2024_finetune_sample_FIXED.csv", 
        "FIXED FINETUNE DATA"
    )
    
    print(f"\n🎯 FINAL VERIFICATION:")
    print(f"Pretrain: Train={train1:.4f}, Test={test1:.4f}")
    print(f"Finetune: Train={train2:.4f}, Test={test2:.4f}")
    
except Exception as e:
    print(f"❌ Error during testing: {e}")