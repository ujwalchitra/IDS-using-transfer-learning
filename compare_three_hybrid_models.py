import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, LSTM, Bidirectional, Dense, 
                                   Dropout, Input, Flatten, GlobalAveragePooling1D,
                                   MultiHeadAttention, LayerNormalization, Embedding)
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
import psutil
import gc

class TransformerBlock(tf.keras.layers.Layer):
    """Transformer Block for BERT-like architecture"""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

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
    
    # Reshape data for 1D models (samples, sequence_length, features)
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

def create_hybrid_bert_cnn_lstm(input_shape, num_classes):
    """Create Hybrid BERT + CNN + LSTM model"""
    print("🔄 BUILDING HYBRID BERT + CNN + LSTM MODEL")
    
    inputs = Input(shape=input_shape)
    
    # CNN Feature Extraction
    cnn1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    cnn1 = MaxPooling1D(pool_size=2)(cnn1)
    cnn1 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(cnn1)
    
    # BERT-like Transformer Blocks
    # Project to transformer dimension
    projected = Dense(128)(cnn1)
    
    # Transformer blocks (simplified BERT)
    transformer1 = TransformerBlock(embed_dim=128, num_heads=8, ff_dim=256)(projected)
    transformer1 = Dropout(0.1)(transformer1)
    
    transformer2 = TransformerBlock(embed_dim=128, num_heads=8, ff_dim=256)(transformer1)
    transformer2 = Dropout(0.1)(transformer2)
    
    # LSTM for sequential processing
    lstm_out = LSTM(128, return_sequences=True)(transformer2)
    lstm_out = LSTM(64)(lstm_out)
    
    # Dense layers
    x = Dense(128, activation='relu')(lstm_out)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=0.0005),  # Lower LR for transformer
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Train and evaluate a single model with comprehensive metrics"""
    print(f"\n🚀 TRAINING {model_name}")
    print("-" * 50)
    
    # Memory before training
    memory_before = get_memory_usage()
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, monitor='val_accuracy'),
        tf.keras.callbacks.ReduceLROnPlateau(patience=8, factor=0.5, min_lr=0.00001)
    ]
    
    # Training time
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,  # More epochs for complex models
        batch_size=32,
        verbose=1,
        callbacks=callbacks
    )
    
    training_time = time.time() - start_time
    
    # Memory after training
    memory_after = get_memory_usage()
    memory_used = memory_after - memory_before
    
    # Evaluation
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Model complexity
    trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params
    
    # FLOPs estimation (simplified)
    flops_estimate = estimate_model_flops(model, X_test.shape)
    
    # Inference time
    start_inference = time.time()
    _ = model.predict(X_test[:100], verbose=0)  # Predict on 100 samples
    inference_time = (time.time() - start_inference) / 100  # Average per sample
    
    # CPU/GPU usage (simplified)
    if tf.config.list_physical_devices('GPU'):
        device = "GPU"
    else:
        device = "CPU"
    
    print(f"✅ {model_name} Results:")
    print(f"   📊 Test Accuracy: {test_accuracy:.4f}")
    print(f"   📉 Test Loss: {test_loss:.4f}")
    print(f"   ⏱️  Training Time: {training_time:.2f} seconds")
    print(f"   ⚡ Inference Time: {inference_time*1000:.2f} ms per sample")
    print(f"   💾 Total Parameters: {total_params:,}")
    print(f"   🎯 Trainable Parameters: {trainable_params:,}")
    print(f"   🧮 Estimated FLOPs: {flops_estimate/1e6:.2f} M")
    print(f"   📱 Memory Used: {memory_used:.2f} MB")
    print(f"   🔧 Device: {device}")
    
    return {
        'model': model,
        'history': history,
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'training_time': training_time,
        'inference_time': inference_time,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'flops_estimate': flops_estimate,
        'memory_used': memory_used,
        'device': device
    }

def estimate_model_flops(model, input_shape):
    """Estimate FLOPs for model (simplified)"""
    # This is a simplified estimation
    # In practice, use tf.profiler for accurate FLOPs
    total_flops = 0
    
    for layer in model.layers:
        if isinstance(layer, Conv1D):
            # Conv1D FLOPs = output_size * kernel_size * input_channels * output_channels
            output_size = input_shape[1] // (2 ** 3)  # Approximate after pooling
            kernel_size = layer.kernel_size[0]
            input_channels = layer.input_shape[-1]
            output_channels = layer.filters
            layer_flops = output_size * kernel_size * input_channels * output_channels
            total_flops += layer_flops
            
        elif isinstance(layer, (LSTM, Bidirectional)):
            # LSTM FLOPs ≈ 4 * (input_size + hidden_size) * hidden_size * sequence_length
            if hasattr(layer, 'units'):
                hidden_size = layer.units
                input_size = layer.input_shape[-1]
                sequence_length = input_shape[1]
                layer_flops = 4 * (input_size + hidden_size) * hidden_size * sequence_length
                if isinstance(layer, Bidirectional):
                    layer_flops *= 2  # Bidirectional doubles computation
                total_flops += layer_flops
                
        elif isinstance(layer, Dense):
            # Dense FLOPs = input_size * output_size
            input_size = layer.input_shape[-1]
            output_size = layer.units
            layer_flops = input_size * output_size
            total_flops += layer_flops
            
        elif 'TransformerBlock' in str(type(layer)):
            # Transformer FLOPs estimation
            total_flops += 1000000  # Simplified estimate
    
    return total_flops

def compare_all_models():
    """Main function to compare all three hybrid models"""
    print("🔬 COMPREHENSIVE HYBRID MODEL COMPARISON")
    print("=" * 70)
    print("📊 Comparing: 1D-CNN+LSTM vs 1D-CNN+BiLSTM vs BERT+CNN+LSTM")
    print("=" * 70)
    
    # Load data
    X_train, X_test, y_train, y_test, num_classes, label_encoder = load_and_prepare_data()
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # Create all models
    print("\n🏗️  BUILDING ALL MODELS")
    print("=" * 50)
    
    cnn_lstm_model = create_hybrid_cnn_lstm(input_shape, num_classes)
    cnn_bilstm_model = create_hybrid_cnn_bilstm(input_shape, num_classes)
    bert_cnn_lstm_model = create_hybrid_bert_cnn_lstm(input_shape, num_classes)
    
    # Print model architectures
    print("\n📐 MODEL ARCHITECTURES SUMMARY:")
    print(f"1D-CNN + LSTM: {cnn_lstm_model.count_params():,} parameters")
    print(f"1D-CNN + BiLSTM: {cnn_bilstm_model.count_params():,} parameters") 
    print(f"BERT + CNN + LSTM: {bert_cnn_lstm_model.count_params():,} parameters")
    
    # Train and evaluate all models
    print("\n" + "="*70)
    print("🎯 MODEL TRAINING AND EVALUATION")
    print("="*70)
    
    results = {}
    
    results['CNN_LSTM'] = train_and_evaluate_model(
        cnn_lstm_model, X_train, X_test, y_train, y_test, "1D-CNN + LSTM"
    )
    
    # Clear memory between trainings
    gc.collect()
    tf.keras.backend.clear_session()
    
    results['CNN_BiLSTM'] = train_and_evaluate_model(
        cnn_bilstm_model, X_train, X_test, y_train, y_test, "1D-CNN + BiLSTM"
    )
    
    # Clear memory between trainings
    gc.collect()
    tf.keras.backend.clear_session()
    
    results['BERT_CNN_LSTM'] = train_and_evaluate_model(
        bert_cnn_lstm_model, X_train, X_test, y_train, y_test, "BERT + CNN + LSTM"
    )
    
    # Comprehensive Comparison
    print("\n" + "="*70)
    print("📊 COMPREHENSIVE COMPARISON RESULTS")
    print("="*70)
    
    comparison_data = {
        'Metric': [
            'Test Accuracy', 'Test Loss', 'Training Time (s)', 
            'Inference Time (ms)', 'Total Parameters', 'Trainable Parameters',
            'FLOPs (M)', 'Memory Used (MB)', 'Device'
        ],
        '1D-CNN + LSTM': [
            results['CNN_LSTM']['test_accuracy'],
            results['CNN_LSTM']['test_loss'],
            results['CNN_LSTM']['training_time'],
            results['CNN_LSTM']['inference_time'] * 1000,
            results['CNN_LSTM']['total_params'],
            results['CNN_LSTM']['trainable_params'],
            results['CNN_LSTM']['flops_estimate'] / 1e6,
            results['CNN_LSTM']['memory_used'],
            results['CNN_LSTM']['device']
        ],
        '1D-CNN + BiLSTM': [
            results['CNN_BiLSTM']['test_accuracy'],
            results['CNN_BiLSTM']['test_loss'],
            results['CNN_BiLSTM']['training_time'],
            results['CNN_BiLSTM']['inference_time'] * 1000,
            results['CNN_BiLSTM']['total_params'],
            results['CNN_BiLSTM']['trainable_params'],
            results['CNN_BiLSTM']['flops_estimate'] / 1e6,
            results['CNN_BiLSTM']['memory_used'],
            results['CNN_BiLSTM']['device']
        ],
        'BERT + CNN + LSTM': [
            results['BERT_CNN_LSTM']['test_accuracy'],
            results['BERT_CNN_LSTM']['test_loss'],
            results['BERT_CNN_LSTM']['training_time'],
            results['BERT_CNN_LSTM']['inference_time'] * 1000,
            results['BERT_CNN_LSTM']['total_params'],
            results['BERT_CNN_LSTM']['trainable_params'],
            results['BERT_CNN_LSTM']['flops_estimate'] / 1e6,
            results['BERT_CNN_LSTM']['memory_used'],
            results['BERT_CNN_LSTM']['device']
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Save models
    cnn_lstm_model.save('04_models/hybrid_cnn_lstm_model.h5')
    cnn_bilstm_model.save('04_models/hybrid_cnn_bilstm_model.h5')
    bert_cnn_lstm_model.save('04_models/hybrid_bert_cnn_lstm_model.h5')
    print(f"\n💾 All models saved to 04_models/")
    
    # Generate comprehensive visualizations
    generate_triple_comparison_plots(results, X_test, y_test, label_encoder)
    
    return results, comparison_df

def generate_triple_comparison_plots(results, X_test, y_test, label_encoder):
    """Generate comprehensive comparison plots for all three models"""
    print("\n📈 GENERATING COMPREHENSIVE COMPARISON VISUALIZATIONS")
    
    # Create results directory
    os.makedirs('05_results', exist_ok=True)
    
    # 1. Performance Metrics Radar Chart
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
    
    # Accuracy comparison
    models = ['1D-CNN + LSTM', '1D-CNN + BiLSTM', 'BERT + CNN + LSTM']
    accuracies = [results['CNN_LSTM']['test_accuracy'], 
                 results['CNN_BiLSTM']['test_accuracy'],
                 results['BERT_CNN_LSTM']['test_accuracy']]
    
    bars = ax1.bar(models, accuracies, color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.7)
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Computational Efficiency (Training Time vs Accuracy)
    training_times = [results['CNN_LSTM']['training_time'],
                     results['CNN_BiLSTM']['training_time'],
                     results['BERT_CNN_LSTM']['training_time']]
    
    scatter = ax2.scatter(training_times, accuracies, s=200, alpha=0.7,
                         c=['skyblue', 'lightcoral', 'lightgreen'])
    ax2.set_xlabel('Training Time (seconds)')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Computational Efficiency: Training Time vs Accuracy', 
                 fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add labels to points
    for i, model in enumerate(models):
        ax2.annotate(model, (training_times[i], accuracies[i]), 
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Memory and Parameter Comparison
    memory_used = [results['CNN_LSTM']['memory_used'],
                  results['CNN_BiLSTM']['memory_used'],
                  results['BERT_CNN_LSTM']['memory_used']]
    
    total_params = [results['CNN_LSTM']['total_params'] / 1e6,
                   results['CNN_BiLSTM']['total_params'] / 1e6,
                   results['BERT_CNN_LSTM']['total_params'] / 1e6]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, memory_used, width, label='Memory Used (MB)', alpha=0.7)
    bars2 = ax3.bar(x + width/2, total_params, width, label='Total Params (Millions)', alpha=0.7)
    ax3.set_xlabel('Models')
    ax3.set_title('Memory and Parameter Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Inference Time Comparison
    inference_times = [results['CNN_LSTM']['inference_time'] * 1000,
                      results['CNN_BiLSTM']['inference_time'] * 1000,
                      results['BERT_CNN_LSTM']['inference_time'] * 1000]
    
    bars = ax4.bar(models, inference_times, color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.7)
    ax4.set_title('Inference Time Comparison', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Inference Time (ms per sample)')
    ax4.set_xlabel('Models')
    for bar, time_val in zip(bars, inference_times):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{time_val:.2f} ms', ha='center', va='bottom')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('05_results/triple_model_comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    print("💾 Saved comprehensive comparison: 05_results/triple_model_comprehensive_comparison.png")
    
    # 2. Training History Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    
    # Accuracy history
    for model_name, result in results.items():
        history = result['history']
        readable_name = model_name.replace('_', ' + ').replace('CNN', '1D-CNN').replace('BERT', 'BERT')
        ax1.plot(history.history['accuracy'], label=f'{readable_name} Train', alpha=0.7)
        ax1.plot(history.history['val_accuracy'], label=f'{readable_name} Val', linestyle='--', alpha=0.7)
    
    ax1.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss history
    for model_name, result in results.items():
        history = result['history']
        readable_name = model_name.replace('_', ' + ').replace('CNN', '1D-CNN').replace('BERT', 'BERT')
        ax2.plot(history.history['loss'], label=f'{readable_name} Train', alpha=0.7)
        ax2.plot(history.history['val_loss'], label=f'{readable_name} Val', linestyle='--', alpha=0.7)
    
    ax2.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('05_results/training_history_comparison.png', dpi=300, bbox_inches='tight')
    print("💾 Saved training history: 05_results/training_history_comparison.png")
    
    # 3. Detailed performance table
    performance_df = pd.DataFrame({
        'Model': models,
        'Accuracy': accuracies,
        'Training Time (s)': training_times,
        'Inference Time (ms)': inference_times,
        'Total Params (M)': total_params,
        'Memory Used (MB)': memory_used,
        'FLOPs (M)': [r['flops_estimate']/1e6 for r in results.values()]
    })
    
    performance_df.to_csv('05_results/detailed_performance_metrics.csv', index=False)
    print("💾 Saved detailed metrics: 05_results/detailed_performance_metrics.csv")
    
    # Print winner analysis
    print("\n" + "="*70)
    print("🏆 PERFORMANCE ANALYSIS AND RECOMMENDATIONS")
    print("="*70)
    
    best_accuracy = max(accuracies)
    best_accuracy_model = models[accuracies.index(best_accuracy)]
    
    fastest_training = min(training_times)
    fastest_model = models[training_times.index(fastest_training)]
    
    fastest_inference = min(inference_times)
    fastest_inference_model = models[inference_times.index(fastest_inference)]
    
    most_efficient = min([mem/acc for mem, acc in zip(memory_used, accuracies)])
    most_efficient_model = models[[mem/acc for mem, acc in zip(memory_used, accuracies)].index(most_efficient)]
    
    print(f"🎯 Most Accurate: {best_accuracy_model} ({best_accuracy:.4f})")
    print(f"⚡ Fastest Training: {fastest_model} ({fastest_training:.2f}s)")
    print(f"🚀 Fastest Inference: {fastest_inference_model} ({fastest_inference:.2f}ms)")
    print(f"💡 Most Efficient (Memory/Accuracy): {most_efficient_model}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if best_accuracy_model == fastest_model:
        print("   ✅ Best model is both accurate and fast!")
    else:
        print("   ⚖️  Trade-off between accuracy and speed")
    
    if "BERT" in best_accuracy_model:
        print("   🔬 BERT-based model shows best accuracy but higher computational cost")
    elif "BiLSTM" in best_accuracy_model:
        print("   🔄 BiLSTM provides good balance of accuracy and computation")
    else:
        print("   ⚡ Simple LSTM offers best computational efficiency")

if __name__ == "__main__":
    # Run the comprehensive comparison
    results, comparison_df = compare_all_models()
    
    print("\n🎉 COMPREHENSIVE COMPARISON COMPLETED!")
    print("📊 All results saved in 05_results/")
    print("💾 All models saved in 04_models/")
    print("📈 Visualizations generated for analysis")