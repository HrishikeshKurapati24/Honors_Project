import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from model import DeepTTC_Model
from utils import BPE_Encoder, DeepTTC_Dataset, load_formatted_data
import argparse

def evaluate(model, device, loader):
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for v_d, v_mask, v_p, y in loader:
            v_d, v_mask, v_p = v_d.to(device), v_mask.to(device), v_p.to(device)
            output = model(v_d, v_mask, v_p)
            
            probs = torch.sigmoid(output).cpu().numpy()
            y_pred.extend(probs)
            y_true.extend(y.numpy())
            
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_class = (y_pred > 0.5).astype(int)
    
    metrics = {
        'AUC': roc_auc_score(y_true, y_pred),
        'Accuracy': accuracy_score(y_true, y_class),
        'F1': f1_score(y_true, y_class),
        'Precision': precision_score(y_true, y_class),
        'Recall': recall_score(y_true, y_class)
    }
    
    cm = confusion_matrix(y_true, y_class)
    return metrics, cm

def main():
    parser = argparse.ArgumentParser(description='DeepTTC Benchmarking Evaluation')
    parser.add_argument('--data_dir', type=str, default='../benchmark formatted dataset', help='Path to formatted dataset')
    parser.add_argument('--vocab_dir', type=str, default='..', help='Path to DeepTTC-main root (for ESPF)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model checkpoint')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Data
    res_df, exp_df, drug_df = load_formatted_data(args.data_dir)
    encoder = BPE_Encoder(args.vocab_dir)
    dataset = DeepTTC_Dataset(res_df, exp_df, drug_df, encoder)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize Model
    model = DeepTTC_Model().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    print(f"\n--- Evaluating Model: {os.path.basename(args.model_path)} ---")
    metrics, cm = evaluate(model, device, loader)
    
    for k, v in metrics.items():
        print(f"{k:10}: {v:.4f}")
    
    print("\nConfusion Matrix:")
    print(cm)

if __name__ == "__main__":
    main()
