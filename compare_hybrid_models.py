import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, LSTM, Bidirectional, 
                                   Dense, Dropout, Input, Flatten, concatenate)
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os

def load_and_prepare_data():
    """Load and prepare the UT 2024 dataset for model comparison"""
    print("📥 LOADING AND PREPARING DATA")
    print("=" * 50)
    
    # Load your balanced dataset
    df = pd.read_csv('02_processed_data/ut2024_balanced_sample.csv')
    
    # Prepare features and labels
    X = df.drop('Label', axis=1).select_dtypes(include=[np.number])
    y = df['Label']
    
    print(f"📊 Dataset shape: {X.shape}")
    print(f"🏷️  Classes: {y.nunique()}")
    print(f"📋 Class distribution:\n{y.value_counts()}")
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    
    # Reshape data for 1D-CNN (samples, timesteps, features)
    # Since we have flat features, we'll treat them as 1D sequences
    X_reshaped = X.values.reshape(X.shape[0], X.shape[1], 1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale the data
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    X_train_scaled_flat = scaler.fit_transform(X_train_flat)
    X_test_scaled_flat = scaler.transform(X_test_flat)
    
    X_train_scaled = X_train_scaled_flat.reshape(X_train.shape)
    X_test_scaled = X_test_scaled_flat.reshape(X_test.shape)
    
    print(f"📐 Input shape for models: {X_train_scaled.shape}")
    print(f"🎯 Number of classes: {num_classes}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, num_classes, le

def create_hybrid_cnn_lstm(input_shape, num_classes):
    """Create Hybrid 1D-CNN + LSTM model"""
    print("🔄 BUILDING HYBRID 1D-CNN + LSTM MODEL")
    
    inputs = Input(shape=input_shape)
    
    # 1D-CNN Branch
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(x)
    
    # LSTM Branch
    x = LSTM(128, return_sequences=True)(x)
    x = LSTM(64)(x)
    
    # Dense layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_hybrid_cnn_bilstm(input_shape, num_classes):
    """Create Hybrid 1D-CNN + BiLSTM model"""
    print("🔄 BUILDING HYBRID 1D-CNN + BiLSTM MODEL")
    
    inputs = Input(shape=input_shape)
    
    # 1D-CNN Branch
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(x)
    
    # BiLSTM Branch
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Bidirectional(LSTM(64))(x)
    
    # Dense layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Train and evaluate a single model"""
    print(f"\n🚀 TRAINING {model_name}")
    print("-" * 40)
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
    ]
    
    # Training time
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        verbose=1,
        callbacks=callbacks
    )
    
    training_time = time.time() - start_time
    
    # Evaluation
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Memory usage (approximate)
    trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params
    
    # Inference time
    start_inference = time.time()
    _ = model.predict(X_test[:100])  # Predict on 100 samples
    inference_time = (time.time() - start_inference) / 100  # Average per sample
    
    print(f"✅ {model_name} Results:")
    print(f"   📊 Test Accuracy: {test_accuracy:.4f}")
    print(f"   📉 Test Loss: {test_loss:.4f}")
    print(f"   ⏱️  Training Time: {training_time:.2f} seconds")
    print(f"   ⚡ Inference Time: {inference_time*1000:.2f} ms per sample")
    print(f"   💾 Total Parameters: {total_params:,}")
    print(f"   🎯 Trainable Parameters: {trainable_params:,}")
    
    return {
        'model': model,
        'history': history,
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'training_time': training_time,
        'inference_time': inference_time,
        'total_params': total_params,
        'trainable_params': trainable_params
    }

def compare_models():
    """Main function to compare both hybrid models"""
    print("🔬 COMPARING HYBRID 1D-CNN+LSTM vs 1D-CNN+BiLSTM")
    print("=" * 60)
    
    # Load data
    X_train, X_test, y_train, y_test, num_classes, label_encoder = load_and_prepare_data()
    
    input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, features)
    
    # Create models
    cnn_lstm_model = create_hybrid_cnn_lstm(input_shape, num_classes)
    cnn_bilstm_model = create_hybrid_cnn_bilstm(input_shape, num_classes)
    
    # Print model architectures
    print("\n📐 MODEL ARCHITECTURES:")
    print("1D-CNN + LSTM Summary:")
    cnn_lstm_model.summary()
    
    print("\n1D-CNN + BiLSTM Summary:")
    cnn_bilstm_model.summary()
    
    # Train and evaluate models
    print("\n" + "="*60)
    print("🎯 MODEL TRAINING AND EVALUATION")
    print("="*60)
    
    cnn_lstm_results = train_and_evaluate_model(
        cnn_lstm_model, X_train, X_test, y_train, y_test, "1D-CNN + LSTM"
    )
    
    cnn_bilstm_results = train_and_evaluate_model(
        cnn_bilstm_model, X_train, X_test, y_train, y_test, "1D-CNN + BiLSTM"
    )
    
    # Comparison Results
    print("\n" + "="*60)
    print("📊 COMPREHENSIVE COMPARISON RESULTS")
    print("="*60)
    
    comparison_data = {
        'Metric': ['Test Accuracy', 'Test Loss', 'Training Time (s)', 
                  'Inference Time (ms)', 'Total Parameters', 'Trainable Parameters'],
        '1D-CNN + LSTM': [
            cnn_lstm_results['test_accuracy'],
            cnn_lstm_results['test_loss'],
            cnn_lstm_results['training_time'],
            cnn_lstm_results['inference_time'] * 1000,
            cnn_lstm_results['total_params'],
            cnn_lstm_results['trainable_params']
        ],
        '1D-CNN + BiLSTM': [
            cnn_bilstm_results['test_accuracy'],
            cnn_bilstm_results['test_loss'],
            cnn_bilstm_results['training_time'],
            cnn_bilstm_results['inference_time'] * 1000,
            cnn_bilstm_results['total_params'],
            cnn_bilstm_results['trainable_params']
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Save models
    cnn_lstm_model.save('04_models/hybrid_cnn_lstm_model.h5')
    cnn_bilstm_model.save('04_models/hybrid_cnn_bilstm_model.h5')
    print(f"\n💾 Models saved to 04_models/")
    
    # Generate visualizations
    generate_comparison_plots(cnn_lstm_results, cnn_bilstm_results, X_test, y_test, label_encoder)
    
    return cnn_lstm_results, cnn_bilstm_results, comparison_df

def generate_comparison_plots(cnn_lstm_results, cnn_bilstm_results, X_test, y_test, label_encoder):
    """Generate comparison plots and charts"""
    print("\n📈 GENERATING COMPARISON VISUALIZATIONS")
    
    # Create results directory
    os.makedirs('05_results', exist_ok=True)
    
    # 1. Accuracy/Loss comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training history comparison
    axes[0,0].plot(cnn_lstm_results['history'].history['accuracy'], label='CNN+LSTM Train')
    axes[0,0].plot(cnn_lstm_results['history'].history['val_accuracy'], label='CNN+LSTM Val')
    axes[0,0].plot(cnn_bilstm_results['history'].history['accuracy'], label='CNN+BiLSTM Train')
    axes[0,0].plot(cnn_bilstm_results['history'].history['val_accuracy'], label='CNN+BiLSTM Val')
    axes[0,0].set_title('Model Accuracy Comparison')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Accuracy')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    axes[0,1].plot(cnn_lstm_results['history'].history['loss'], label='CNN+LSTM Train')
    axes[0,1].plot(cnn_lstm_results['history'].history['val_loss'], label='CNN+LSTM Val')
    axes[0,1].plot(cnn_bilstm_results['history'].history['loss'], label='CNN+BiLSTM Train')
    axes[0,1].plot(cnn_bilstm_results['history'].history['val_loss'], label='CNN+BiLSTM Val')
    axes[0,1].set_title('Model Loss Comparison')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Loss')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # Performance metrics bar chart
    metrics = ['Accuracy', 'Training Time (s)', 'Inference Time (ms)']
    cnn_lstm_metrics = [cnn_lstm_results['test_accuracy'], 
                       cnn_lstm_results['training_time'],
                       cnn_lstm_results['inference_time'] * 1000]
    cnn_bilstm_metrics = [cnn_bilstm_results['test_accuracy'], 
                         cnn_bilstm_results['training_time'],
                         cnn_bilstm_results['inference_time'] * 1000]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[1,0].bar(x - width/2, cnn_lstm_metrics, width, label='CNN+LSTM', alpha=0.7)
    axes[1,0].bar(x + width/2, cnn_bilstm_metrics, width, label='CNN+BiLSTM', alpha=0.7)
    axes[1,0].set_title('Performance Metrics Comparison')
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels(metrics)
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Parameter comparison
    param_metrics = ['Total Params', 'Trainable Params']
    cnn_lstm_params = [cnn_lstm_results['total_params'] / 1e6, 
                      cnn_lstm_results['trainable_params'] / 1e6]
    cnn_bilstm_params = [cnn_bilstm_results['total_params'] / 1e6, 
                        cnn_bilstm_results['trainable_params'] / 1e6]
    
    x_param = np.arange(len(param_metrics))
    axes[1,1].bar(x_param - width/2, cnn_lstm_params, width, label='CNN+LSTM', alpha=0.7)
    axes[1,1].bar(x_param + width/2, cnn_bilstm_params, width, label='CNN+BiLSTM', alpha=0.7)
    axes[1,1].set_title('Model Parameters (Millions)')
    axes[1,1].set_xticks(x_param)
    axes[1,1].set_xticklabels(param_metrics)
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('05_results/model_comparison_metrics.png', dpi=300, bbox_inches='tight')
    print("💾 Saved comparison metrics: 05_results/model_comparison_metrics.png")
    
    # 2. Confusion matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # CNN+LSTM confusion matrix
    y_pred_lstm = np.argmax(cnn_lstm_results['model'].predict(X_test), axis=1)
    cm_lstm = confusion_matrix(y_test, y_pred_lstm)
    sns.heatmap(cm_lstm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    ax1.set_title('1D-CNN + LSTM - Confusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # CNN+BiLSTM confusion matrix
    y_pred_bilstm = np.argmax(cnn_bilstm_results['model'].predict(X_test), axis=1)
    cm_bilstm = confusion_matrix(y_test, y_pred_bilstm)
    sns.heatmap(cm_bilstm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    ax2.set_title('1D-CNN + BiLSTM - Confusion Matrix')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('05_results/confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
    print("💾 Saved confusion matrices: 05_results/confusion_matrices_comparison.png")
    
    # 3. Detailed classification reports
    print("\n📋 DETAILED CLASSIFICATION REPORTS")
    print("=" * 50)
    print("1D-CNN + LSTM Classification Report:")
    print(classification_report(y_test, y_pred_lstm, target_names=label_encoder.classes_))
    
    print("\n1D-CNN + BiLSTM Classification Report:")
    print(classification_report(y_test, y_pred_bilstm, target_names=label_encoder.classes_))

if __name__ == "__main__":
    # Run the comparison
    cnn_lstm_results, cnn_bilstm_results, comparison_df = compare_models()
    
    print("\n🎉 COMPARISON COMPLETED!")
    print("📊 Results saved in 05_results/")
    print("💾 Models saved in 04_models/")
    
    # Print winner
    if cnn_lstm_results['test_accuracy'] > cnn_bilstm_results['test_accuracy']:
        winner = "1D-CNN + LSTM"
        accuracy_diff = cnn_lstm_results['test_accuracy'] - cnn_bilstm_results['test_accuracy']
    else:
        winner = "1D-CNN + BiLSTM"
        accuracy_diff = cnn_bilstm_results['test_accuracy'] - cnn_lstm_results['test_accuracy']
    
    print(f"\n🏆 WINNER: {winner} (Accuracy difference: {accuracy_diff:.4f})")