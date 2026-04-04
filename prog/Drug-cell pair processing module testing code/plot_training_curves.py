import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def plot_training_curves():
    print("============================================================")
    print("Training Curve Generator")
    print("============================================================")
    
    # Check for metrics files
    metrics_files = glob.glob("metrics_curves_*.npy")
    
    if not metrics_files:
        print("No metrics files found. Make sure to run experiments with --save_training_curves flag first.")
        return

    print(f"Found {len(metrics_files)} metrics file(s)")
    
    # 1. Loss Curves
    plt.figure(figsize=(12, 6))
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    line_styles = ['-', '--', '-.', ':']
    
    for i, f in enumerate(metrics_files):
        try:
            # Parse filename for legend
            # Expected: metrics_curves_heterogenous_SAGE_GTTrue_CL0.1_T0.1.npy
            basename = os.path.basename(f)
            label = basename.replace('metrics_curves_', '').replace('.npy', '')
            
            # Load Data
            metrics = np.load(f, allow_pickle=True).item()
            train_loss = metrics['train_loss']
            
            # Plot
            epochs = range(1, len(train_loss) + 1)
            plt.plot(epochs, train_loss, label=label, color=colors[i % len(colors)], linestyle=line_styles[i % len(line_styles)])
            
        except Exception as e:
            print(f"Error processing {f}: {e}")

    plt.xlabel('Epochs')
    plt.ylabel('Total Training Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)
    
    loss_plot_file = 'training_loss_comparison.png'
    plt.savefig(loss_plot_file)
    print(f"Loss curve plot saved to: {loss_plot_file}")
    plt.close()

    # 2. AUC Curves (Validation)
    plt.figure(figsize=(12, 6))
    
    for i, f in enumerate(metrics_files):
        try:
            basename = os.path.basename(f)
            label = basename.replace('metrics_curves_', '').replace('.npy', '')
            
            metrics = np.load(f, allow_pickle=True).item()
            val_auc = metrics['val_auc']
            
            epochs = range(1, len(val_auc) + 1)
            plt.plot(epochs, val_auc, label=label, color=colors[i % len(colors)], linestyle=line_styles[i % len(line_styles)])
            
        except Exception as e:
            pass

    plt.xlabel('Epochs')
    plt.ylabel('Validation AUC')
    plt.title('Validation AUC over Epochs')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    auc_plot_file = 'training_auc_comparison.png'
    plt.savefig(auc_plot_file)
    print(f"AUC curve plot saved to: {auc_plot_file}")
    plt.close()
    
    # 3. Contrastive Loss specific (if available)
    # Only if CL was actually used (loss > 0)
    plt.figure(figsize=(12, 6))
    has_cl_plots = False
    
    for i, f in enumerate(metrics_files):
        try:
            basename = os.path.basename(f)
            label = basename.replace('metrics_curves_', '').replace('.npy', '')
            
            metrics = np.load(f, allow_pickle=True).item()
            cont_loss = metrics['cont_loss']
            
            if np.sum(cont_loss) > 0:
                has_cl_plots = True
                epochs = range(1, len(cont_loss) + 1)
                plt.plot(epochs, cont_loss, label=label, color=colors[i % len(colors)], linestyle=line_styles[i % len(line_styles)])
            
        except Exception as e:
            pass
            
    if has_cl_plots:
        plt.xlabel('Epochs')
        plt.ylabel('Contrastive Loss')
        plt.title('Contrastive Loss (SupCon) over Epochs')
        plt.legend()
        plt.grid(True)
        
        cl_plot_file = 'contrastive_loss_comparison.png'
        plt.savefig(cl_plot_file)
        print(f"Contrastive Loss curve plot saved to: {cl_plot_file}")
    plt.close()

if __name__ == "__main__":
    plot_training_curves()