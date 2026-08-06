# evaluate_pretrained.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import joblib
import os

print("📊 EVALUATING PRE-TRAINED MODEL ACCURACY...")
print("=" * 50)

# Load the data and artifacts
base_path = "D:/IoT-DDoS-Detection/iot_ddos_detection/"
sequence_data_path = os.path.join(base_path, "03_sequence_data")
models_path = os.path.join(base_path, "04_models")

# Load sequence data
sequence_file = os.path.join(sequence_data_path, "sequence_datasets.npz")
data = np.load(sequence_file, allow_pickle=True)

X_val_2017 = data['X_val']
y_val_2017 = data['y_val']
X_test_2017 = data['X_test'] 
y_test_2017 = data['y_test']
class_names_2017 = data['class_names']

# Convert to categorical
y_val_2017_cat = to_categorical(y_val_2017, num_classes=len(class_names_2017))
y_test_2017_cat = to_categorical(y_test_2017, num_classes=len(class_names_2017))

# Load the pre-trained model
model_path = os.path.join(models_path, 'pretrained_cic2017_model.h5')
best_model = tf.keras.models.load_model(model_path)

# Evaluate on validation set
val_loss, val_accuracy, val_precision, val_recall = best_model.evaluate(
    X_val_2017, y_val_2017_cat, verbose=0
)

# Evaluate on test set
test_loss, test_accuracy, test_precision, test_recall = best_model.evaluate(
    X_test_2017, y_test_2017_cat, verbose=0
)

print("🎯 PRE-TRAINED MODEL ACCURACY RESULTS:")
print("=" * 40)
print(f"✅ VALIDATION SET:")
print(f"   📈 Accuracy:    {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
print(f"   🎯 Precision:   {val_precision:.4f}")
print(f"   🔄 Recall:      {val_recall:.4f}")
print(f"   📉 Loss:        {val_loss:.4f}")

print(f"\n✅ TEST SET:")
print(f"   📈 Accuracy:    {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"   🎯 Precision:   {test_precision:.4f}")
print(f"   🔄 Recall:      {test_recall:.4f}")
print(f"   📉 Loss:        {test_loss:.4f}")

# Calculate F1-score
val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall)
test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)

print(f"\n📊 ADDITIONAL METRICS:")
print(f"   🎯 Validation F1-Score: {val_f1:.4f}")
print(f"   🎯 Test F1-Score:       {test_f1:.4f}")

# Performance interpretation
print(f"\n📋 PERFORMANCE INTERPRETATION:")
if val_accuracy >= 0.95:
    print("   🏆 EXCELLENT: Model shows outstanding performance!")
elif val_accuracy >= 0.90:
    print("   ✅ VERY GOOD: Strong detection capabilities!")
elif val_accuracy >= 0.85:
    print("   👍 GOOD: Solid performance for intrusion detection!")
elif val_accuracy >= 0.80:
    print("   ⚠️  FAIR: Acceptable but could be improved!")
else:
    print("   🔄 NEEDS IMPROVEMENT: Consider tuning hyperparameters!")

print(f"\n🎯 READY FOR FINE-TUNING:")
print(f"   The pre-trained model achieved {val_accuracy*100:.2f}% accuracy")
print(f"   This provides a strong foundation for transfer learning to CIC-IDS2018!")