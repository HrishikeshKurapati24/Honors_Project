#!/usr/bin/env python3
"""
ROC Curve Generator for Drug Response Prediction Models
Reads y_true and y_pred .npy files and generates ROC-AUC curves.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import glob
import os

def plot_roc_curves():
    """Generate ROC curves from all available prediction files."""
    
    # Find all y_pred files
    pred_files = glob.glob("y_pred_*.npy")
    
    if not pred_files:
        print("No prediction files found. Make sure to run experiments with --save_predictions flag first.")
        return
    
    print(f"Found {len(pred_files)} prediction file(s)")
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Model labels for legend (more readable than filenames)
    model_labels = {
        'heterogenous_SAGE_GTFalse_CL0.01_T0.05': 'Proposed (Best) - Hetero+SAGE+CL',
        'heterogenous_SAGE_GTFalse_CL0.1_T0.1': 'Proposed - Hetero+SAGE (No CL)',
        'homogenous_GCN_GTTrue_CL0.1_T0.1': 'Baseline - Homo+GCN+GT'
    }
    
    roc_data = []
    
    for pred_file in sorted(pred_files):
        # Load predictions and labels
        y_pred = np.load(pred_file)
        
        # Construct corresponding y_true filename
        true_file = pred_file.replace('y_pred_', 'y_true_')
        
        if not os.path.exists(true_file):
            print(f"Warning: Missing {true_file} for {pred_file}, skipping...")
            continue
        
        y_true = np.load(true_file)
        
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        # Extract model config from filename
        config = pred_file.replace('y_pred_preds_', '').replace('.npy', '')
        
        # Use readable label if available
        label = model_labels.get(config, config)
        label = f"{label} (AUC = {roc_auc:.4f})"
        
        # Plot ROC curve
        plt.plot(fpr, tpr, linewidth=2, label=label)
        
        roc_data.append({
            'config': config,
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc
        })
        
        print(f"Model: {config}")
        print(f"  AUC-ROC: {roc_auc:.4f}")
        print(f"  Samples: {len(y_true)}")
        print()
    
    # Plot diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5000)')
    
    # Formatting
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Drug Response Prediction Models', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    output_file = 'roc_curves_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"ROC curve plot saved to: {output_file}")
    
    # Also save as PDF for publication quality
    pdf_file = 'roc_curves_comparison.pdf'
    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"PDF version saved to: {pdf_file}")
    
    plt.show()
    
    return roc_data

if __name__ == "__main__":
    print("="*60)
    print("ROC Curve Generator")
    print("="*60)
    print()
    
    roc_data = plot_roc_curves()
    
    if roc_data:
        print("\nSummary:")
        print("-" * 60)
        sorted_models = sorted(roc_data, key=lambda x: x['auc'], reverse=True)
        for i, model in enumerate(sorted_models, 1):
            print(f"{i}. {model['config']}: AUC = {model['auc']:.4f}")