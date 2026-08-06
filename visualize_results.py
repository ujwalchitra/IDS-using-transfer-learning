# visualize_results.py - Generate graphs and visualizations
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os
import glob
from datetime import datetime

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

class ResultsVisualizer:
    def __init__(self):
        self.results_dir = 'results'
        self.figures_dir = 'figures'
        os.makedirs(self.figures_dir, exist_ok=True)
        
        # Load your results
        self.results = {
            "1DCNN_LSTM": {
                "model_name": "1DCNN_LSTM",
                "parameters": 41954,
                "pretrain_samples": 278644,
                "finetune_samples": 138907,
                "best_pretrain_acc": 0.9833300436038688,
                "best_finetune_acc": 0.9930530559354978,
                "improvement": 0.009723012331628977,
                "total_time": 547.8481874465942
            },
            "BERT_CNN_LSTM": {
                "model_name": "BERT_CNN_LSTM",
                "parameters": 42802,
                "pretrain_samples": 278644,
                "finetune_samples": 138907,
                "best_pretrain_acc": 0.9127563745985035,
                "best_finetune_acc": 0.8267943272622561,
                "improvement": -0.08596204733624735,
                "total_time": 423.11046528816223
            },
            "1DCNN_BiLSTM": {
                "model_name": "1DCNN_BiLSTM",
                "parameters": 33762,
                "pretrain_samples": 278644,
                "finetune_samples": 138907,
                "best_pretrain_acc": 0.9862010802275296,
                "best_finetune_acc": 0.9947088042617522,
                "improvement": 0.008507724034222619,
                "total_time": 533.232931137085
            }
        }
    
    def create_comparison_plot(self):
        """Create main comparison plot"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Transfer Learning Model Comparison for IoT DDoS Detection', fontsize=16, fontweight='bold')
        
        models = list(self.results.keys())
        pretrain_acc = [self.results[m]['best_pretrain_acc'] for m in models]
        finetune_acc = [self.results[m]['best_finetune_acc'] for m in models]
        improvements = [self.results[m]['improvement'] for m in models]
        parameters = [self.results[m]['parameters'] for m in models]
        times = [self.results[m]['total_time']/60 for m in models]  # Convert to minutes
        
        # Plot 1: Accuracy Comparison
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, pretrain_acc, width, label='Pretrain Accuracy', alpha=0.8)
        bars2 = ax1.bar(x + width/2, finetune_acc, width, label='Finetune Accuracy', alpha=0.8)
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy: Pretraining vs Fine-tuning')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # Plot 2: Improvement after Fine-tuning
        colors = ['green' if x >= 0 else 'red' for x in improvements]
        bars = ax2.bar(models, improvements, color=colors, alpha=0.7)
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Accuracy Improvement')
        ax2.set_title('Improvement After Fine-tuning')
        ax2.set_xticklabels(models, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:+.3f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        # Plot 3: Model Size vs Performance
        scatter = ax3.scatter(parameters, finetune_acc, s=100, alpha=0.7)
        ax3.set_xlabel('Number of Parameters')
        ax3.set_ylabel('Final Accuracy')
        ax3.set_title('Model Size vs Performance')
        ax3.grid(True, alpha=0.3)
        
        # Add model labels to scatter points
        for i, model in enumerate(models):
            ax3.annotate(model, (parameters[i], finetune_acc[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        # Plot 4: Training Time vs Performance
        bars = ax4.bar(models, times, alpha=0.7)
        ax4.set_xlabel('Models')
        ax4.set_ylabel('Training Time (minutes)')
        ax4.set_title('Training Time Comparison')
        ax4.set_xticklabels(models, rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}m', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{self.figures_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.figures_dir}/model_comparison.pdf', bbox_inches='tight')
        plt.show()
        
        return fig
    
    def create_performance_radar(self):
        """Create radar chart for model performance"""
        fig = plt.figure(figsize=(10, 8))
        
        # Normalize metrics for radar chart
        models = list(self.results.keys())
        
        # Metrics to compare
        categories = ['Accuracy', 'Improvement', 'Efficiency', 'Speed']
        
        # Normalize values (0-1 scale)
        accuracy_norm = [self.results[m]['best_finetune_acc'] for m in models]
        improvement_norm = [(self.results[m]['improvement'] + 0.1) / 0.2 for m in models]  # Normalize -0.1 to 0.1
        efficiency_norm = [1 - (self.results[m]['parameters'] / 50000) for m in models]  # Inverse of parameter count
        speed_norm = [1 - (self.results[m]['total_time'] / 600) for m in models]  # Inverse of time
        
        # Set angles for radar chart
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Create radar chart
        ax = fig.add_subplot(111, polar=True)
        
        for i, model in enumerate(models):
            values = [accuracy_norm[i], improvement_norm[i], efficiency_norm[i], speed_norm[i]]
            values += values[:1]  # Complete the circle
            ax.plot(angles, values, 'o-', linewidth=2, label=model)
            ax.fill(angles, values, alpha=0.1)
        
        # Add category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        
        # Add grid
        ax.grid(True)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.title('Model Performance Radar Chart', size=14, fontweight='bold')
        plt.savefig(f'{self.figures_dir}/performance_radar.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def create_ranking_plot(self):
        """Create model ranking plot"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        models = list(self.results.keys())
        
        # Calculate overall score (weighted combination of metrics)
        scores = []
        for model in models:
            accuracy_score = self.results[model]['best_finetune_acc'] * 0.4
            improvement_score = (self.results[model]['improvement'] + 0.1) * 2.5 * 0.3  # Normalize and weight
            efficiency_score = (1 - self.results[model]['parameters'] / 50000) * 0.2
            speed_score = (1 - self.results[model]['total_time'] / 600) * 0.1
            total_score = accuracy_score + improvement_score + efficiency_score + speed_score
            scores.append(total_score)
        
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]
        sorted_models = [models[i] for i in sorted_indices]
        sorted_scores = [scores[i] for i in sorted_indices]
        
        # Create ranking plot
        bars = ax.barh(sorted_models, sorted_scores, color=sns.color_palette("viridis", len(models)))
        ax.set_xlabel('Overall Performance Score')
        ax.set_title('Model Ranking - Overall Performance', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{width:.3f}', ha='left', va='center')
        
        # Add performance details as text
        performance_text = "\n".join([
            f"🏆 Best Accuracy: {sorted_models[0]} ({self.results[sorted_models[0]]['best_finetune_acc']:.3f})",
            f"⚡ Most Efficient: {sorted_models[0]} ({self.results[sorted_models[0]]['parameters']:,} params)",
            f"📈 Best Improvement: {sorted_models[0]} (+{self.results[sorted_models[0]]['improvement']:.3f})"
        ])
        
        ax.text(0.02, 0.98, performance_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{self.figures_dir}/model_ranking.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def create_summary_table(self):
        """Create a summary table image"""
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        table_data = []
        headers = ['Model', 'Parameters', 'Pretrain Acc', 'Finetune Acc', 
                  'Improvement', 'Time (min)', 'Status']
        
        for model, data in self.results.items():
            status = "✅ Excellent" if data['improvement'] > 0 and data['best_finetune_acc'] > 0.98 else "⚠️ Good" if data['best_finetune_acc'] > 0.95 else "❌ Needs Work"
            
            table_data.append([
                model,
                f"{data['parameters']:,}",
                f"{data['best_pretrain_acc']:.3f}",
                f"{data['best_finetune_acc']:.3f}",
                f"{data['improvement']:+.3f}",
                f"{data['total_time']/60:.1f}",
                status
            ])
        
        # Create table
        table = ax.table(cellText=table_data, colLabels=headers,
                        loc='center', cellLoc='center')
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Color cells based on performance
        for i, (model, data) in enumerate(self.results.items()):
            if data['improvement'] > 0:
                table[(i+1, 4)].set_facecolor('#90EE90')  # Green for positive improvement
            else:
                table[(i+1, 4)].set_facecolor('#FFB6C1')  # Red for negative improvement
            
            if data['best_finetune_acc'] > 0.98:
                table[(i+1, 3)].set_facecolor('#90EE90')  # Green for high accuracy
            elif data['best_finetune_acc'] > 0.95:
                table[(i+1, 3)].set_facecolor('#FFFACD')  # Yellow for medium accuracy
        
        plt.title('Transfer Learning Results Summary', fontweight='bold', fontsize=14)
        plt.savefig(f'{self.figures_dir}/summary_table.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def generate_analysis_report(self):
        """Generate a text analysis report"""
        print("="*70)
        print("📊 TRANSFER LEARNING ANALYSIS REPORT")
        print("="*70)
        
        best_model = max(self.results.items(), key=lambda x: x[1]['best_finetune_acc'])
        most_efficient = min(self.results.items(), key=lambda x: x[1]['parameters'])
        fastest = min(self.results.items(), key=lambda x: x[1]['total_time'])
        
        print(f"\n🏆 BEST PERFORMING MODEL: {best_model[0]}")
        print(f"   Final Accuracy: {best_model[1]['best_finetune_acc']:.3f}")
        print(f"   Improvement: +{best_model[1]['improvement']:.3f}")
        print(f"   Parameters: {best_model[1]['parameters']:,}")
        
        print(f"\n⚡ MOST EFFICIENT MODEL: {most_efficient[0]}")
        print(f"   Parameters: {most_efficient[1]['parameters']:,}")
        print(f"   Accuracy: {most_efficient[1]['best_finetune_acc']:.3f}")
        
        print(f"\n🚀 FASTEST TRAINING: {fastest[0]}")
        print(f"   Time: {fastest[1]['total_time']/60:.1f} minutes")
        print(f"   Accuracy: {fastest[1]['best_finetune_acc']:.3f}")
        
        print(f"\n📈 KEY INSIGHTS:")
        print(f"   • All models achieved >98% accuracy except BERT_CNN_LSTM")
        print(f"   • 1DCNN_BiLSTM has the highest accuracy (99.47%)")
        print(f"   • BERT_CNN_LSTM showed negative transfer learning")
        print(f"   • 1DCNN_BiLSTM is both accurate and parameter-efficient")
        
        # Save report to file
        with open(f'{self.figures_dir}/analysis_report.txt', 'w') as f:
            f.write("Transfer Learning Analysis Report\n")
            f.write("="*50 + "\n")
            f.write(f"Best Model: {best_model[0]} (Accuracy: {best_model[1]['best_finetune_acc']:.3f})\n")
            f.write(f"Most Efficient: {most_efficient[0]} ({most_efficient[1]['parameters']:,} params)\n")
            f.write(f"Fastest: {fastest[0]} ({fastest[1]['total_time']/60:.1f} minutes)\n")
    
    def create_all_visualizations(self):
        """Generate all visualizations"""
        print("🎨 Generating visualizations...")
        
        self.create_comparison_plot()
        print("✓ Created comparison plot")
        
        self.create_performance_radar()
        print("✓ Created performance radar chart")
        
        self.create_ranking_plot()
        print("✓ Created model ranking plot")
        
        self.create_summary_table()
        print("✓ Created summary table")
        
        self.generate_analysis_report()
        print("✓ Generated analysis report")
        
        print(f"\n✅ All visualizations saved to '{self.figures_dir}' directory")
        print("📊 Files created:")
        print("   - model_comparison.png/pdf")
        print("   - performance_radar.png")
        print("   - model_ranking.png")
        print("   - summary_table.png")
        print("   - analysis_report.txt")

if __name__ == "__main__":
    visualizer = ResultsVisualizer()
    visualizer.create_all_visualizations()