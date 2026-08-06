import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import os

def create_sequential_model(input_shape, num_classes):
    """Create a new model that matches your data shape"""
    print("🆕 CREATING NEW MODEL ARCHITECTURE")
    print(f"📐 Input shape: {input_shape}")
    print(f"🎯 Output classes: {num_classes}")
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def transfer_weights(old_model, new_model):
    """Transfer weights from pre-trained model where possible"""
    print("🔄 TRANSFERRING WEIGHTS FROM PRE-TRAINED MODEL")
    
    # Get dense layers from old model
    old_dense_layers = [layer for layer in old_model.layers 
                       if isinstance(layer, tf.keras.layers.Dense)]
    
    # Get dense layers from new model
    new_dense_layers = [layer for layer in new_model.layers 
                       if isinstance(layer, tf.keras.layers.Dense)]
    
    # Transfer weights for matching layers
    transferred = 0
    for i, (old_layer, new_layer) in enumerate(zip(old_dense_layers, new_dense_layers)):
        try:
            # Check if weight shapes match
            if (i < len(new_dense_layers) - 1 and  # Don't transfer output layer
                old_layer.weights[0].shape == new_layer.weights[0].shape):
                
                new_layer.set_weights(old_layer.get_weights())
                print(f"   ✅ Transferred weights: {old_layer.name} -> {new_layer.name}")
                transferred += 1
            else:
                print(f"   ⚠️  Skipped {old_layer.name} (shape mismatch)")
                
        except Exception as e:
            print(f"   ❌ Error transferring {old_layer.name}: {e}")
    
    print(f"🔢 Transferred weights for {transferred} layers")
    return new_model

def quick_finetune_fixed():
    print("🎯 FINE-TUNING WITH FIXED ARCHITECTURE")
    print("=" * 50)
    
    # Load the balanced sample
    sample_path = '02_processed_data/ut2024_balanced_sample.csv'
    
    if not os.path.exists(sample_path):
        print(f"❌ Sample file not found: {sample_path}")
        return
    
    print(f"📥 Loading: {sample_path}")
    df = pd.read_csv(sample_path)
    
    print(f"📊 Dataset: {df.shape[0]} samples, {df.shape[1]} features")
    
    # Prepare features and labels
    X = df.drop('Label', axis=1).select_dtypes(include=[np.number])
    y = df['Label']
    
    print(f"🔢 Features: {X.shape[1]}")
    print(f"🏷️  Labels: {y.nunique()} classes")
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    
    print(f"🎯 Encoded classes: {num_classes}")
    print(f"📊 Class distribution: {dict(zip(le.classes_, np.bincount(y_encoded)))}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n📚 Training samples: {X_train_scaled.shape[0]}")
    print(f"🧪 Test samples: {X_test_scaled.shape[0]}")
    print(f"📐 Input shape: {X_train_scaled.shape[1]}")
    
    # Strategy 1: Create new model with correct input shape
    input_shape = (X_train_scaled.shape[1],)
    
    # Try to load pre-trained model for weight transfer
    pretrained_model = None
    model_files = [f for f in os.listdir('04_models/') if f.endswith('.h5')]
    
    if model_files:
        model_path = f'04_models/{model_files[0]}'
        try:
            pretrained_model = tf.keras.models.load_model(model_path)
            print(f"✅ Loaded pre-trained model: {model_files[0]}")
        except Exception as e:
            print(f"⚠️  Could not load pre-trained model: {e}")
            pretrained_model = None
    
    # Create new model
    if pretrained_model:
        print("\n🔄 USING TRANSFER LEARNING APPROACH")
        # Create model with same architecture as pre-trained but correct input shape
        new_model = create_sequential_model(input_shape, num_classes)
        new_model = transfer_weights(pretrained_model, new_model)
    else:
        print("\n🆕 TRAINING FROM SCRATCH")
        new_model = create_sequential_model(input_shape, num_classes)
    
    # Train the model
    print("\n🚀 STARTING TRAINING...")
    history = new_model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=50,
        batch_size=16,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=10, 
                restore_best_weights=True,
                monitor='val_accuracy'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                patience=5,
                factor=0.5,
                min_lr=0.00001
            )
        ]
    )
    
    # Evaluate
    print("\n📊 FINAL EVALUATION:")
    test_loss, test_acc = new_model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"🎯 Test Accuracy: {test_acc:.4f}")
    print(f"📉 Test Loss: {test_loss:.4f}")
    
    # Save model
    output_path = '04_models/model_finetuned_ut2024_fixed.h5'
    new_model.save(output_path)
    print(f"💾 Saved model: {output_path}")
    
    # Show predictions
    y_pred = np.argmax(new_model.predict(X_test_scaled), axis=1)
    
    from sklearn.metrics import classification_report
    print("\n📋 CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    return new_model, test_acc

if __name__ == "__main__":
    quick_finetune_fixed()