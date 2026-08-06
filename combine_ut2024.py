import pandas as pd
import os
import numpy as np
from sklearn.utils import resample

def create_finetune_sample(sample_size=500000):
    """Create a balanced sample for fine-tuning directly from individual files"""
    print("🎯 CREATING SAMPLE FOR FINE-TUNING")
    print("=" * 50)
    
    ut2024_path = 'UT_2024'
    csv_files = [f for f in os.listdir(ut2024_path) if f.endswith('.csv')]
    
    print(f"📁 Found {len(csv_files)} CSV files")
    
    # Sample from each file
    samples_per_file = sample_size // len(csv_files)
    all_samples = []
    
    for file in csv_files:
        file_path = os.path.join(ut2024_path, file)
        print(f"\n📥 Sampling from: {file}")
        
        try:
            # Read a sample from each file
            df_sample = pd.read_csv(file_path, nrows=samples_per_file * 3)  # Read extra for cleaning
            
            # Basic cleaning
            df_clean = df_sample.dropna().drop_duplicates()
            
            # Take final sample
            if len(df_clean) > samples_per_file:
                final_sample = df_clean.sample(n=samples_per_file, random_state=42)
            else:
                final_sample = df_clean
            
            print(f"   ✅ Sampled: {final_sample.shape[0]} rows")
            all_samples.append(final_sample)
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    if not all_samples:
        print("❌ No samples collected!")
        return None
    
    # Combine all samples
    combined_sample = pd.concat(all_samples, ignore_index=True)
    print(f"\n📊 COMBINED SAMPLE: {combined_sample.shape[0]} rows, {combined_sample.shape[1]} columns")
    
    # Remove any remaining duplicates
    combined_sample = combined_sample.drop_duplicates()
    print(f"📊 AFTER DEDUPLICATION: {combined_sample.shape[0]} rows")
    
    # Save the sample
    output_path = '02_processed_data/ut2024_finetune_sample.csv'
    os.makedirs('02_processed_data', exist_ok=True)
    
    combined_sample.to_csv(output_path, index=False)
    print(f"💾 Saved: {output_path}")
    
    # Show dataset info
    print_dataset_info(combined_sample)
    
    return combined_sample

def print_dataset_info(df):
    """Print dataset information"""
    print("\n📈 DATASET INFO:")
    print(f"   - Total samples: {df.shape[0]:,}")
    print(f"   - Total features: {df.shape[1]}")
    
    if 'Label' in df.columns:
        label_counts = df['Label'].value_counts()
        print(f"   - Label distribution:")
        for label, count in label_counts.items():
            percentage = count / len(df) * 100
            print(f"        {label}: {count:,} samples ({percentage:.1f}%)")
    
    # Memory usage
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"   - Memory usage: {memory_mb:.2f} MB")

def create_balanced_sample():
    """Create a balanced sample with equal class distribution"""
    print("\n⚖️ CREATING BALANCED SAMPLE")
    
    try:
        # Load the sample we just created
        sample_path = '02_processed_data/ut2024_finetune_sample.csv'
        df = pd.read_csv(sample_path)
        
        if 'Label' not in df.columns:
            print("❌ No 'Label' column found")
            return df
        
        # Find the minimum class count
        min_class_count = df['Label'].value_counts().min()
        balanced_size = min(min_class_count * df['Label'].nunique(), 200000)  # Max 200K
        
        print(f"🏷️  Balancing classes (min: {min_class_count} per class)")
        
        # Create balanced sample
        balanced_samples = []
        for label in df['Label'].unique():
            label_data = df[df['Label'] == label]
            sample_size = min(len(label_data), balanced_size // df['Label'].nunique())
            balanced_sample = label_data.sample(n=sample_size, random_state=42)
            balanced_samples.append(balanced_sample)
        
        balanced_df = pd.concat(balanced_samples, ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
        
        print(f"📊 BALANCED SAMPLE: {balanced_df.shape[0]} rows")
        print(f"🏷️  Final distribution:")
        print(balanced_df['Label'].value_counts())
        
        # Save balanced sample
        balanced_path = '02_processed_data/ut2024_balanced_sample.csv'
        balanced_df.to_csv(balanced_path, index=False)
        print(f"💾 Saved balanced sample: {balanced_path}")
        
        return balanced_df
        
    except Exception as e:
        print(f"❌ Error creating balanced sample: {e}")
        return None

if __name__ == "__main__":
    # Create sample for fine-tuning
    sample = create_finetune_sample(300000)  # 300K samples
    
    if sample is not None:
        # Create balanced version
        balanced_sample = create_balanced_sample()
        
        print(f"\n✅ READY FOR FINE-TUNING!")
        print(f"📁 Use: 02_processed_data/ut2024_finetune_sample.csv")
        if balanced_sample is not None:
            print(f"⚖️  Or: 02_processed_data/ut2024_balanced_sample.csv (balanced)")
        
        print(f"\n🎯 Next: Run your fine-tuning script on these samples!")