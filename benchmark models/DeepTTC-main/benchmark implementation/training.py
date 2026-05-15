import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from model import DeepTTC_Model
from utils import BPE_Encoder, DeepTTC_Dataset, load_formatted_data
import argparse

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (v_d, v_mask, v_p, y) in enumerate(train_loader):
        v_d, v_mask, v_p, y = v_d.to(device), v_mask.to(device), v_p.to(device), y.to(device)
        
        optimizer.zero_grad()
        output = model(v_d, v_mask, v_p)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(v_d)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}")
    
    return total_loss / len(train_loader)

def evaluate(model, device, loader, criterion):
    model.eval()
    val_loss = 0
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for v_d, v_mask, v_p, y in loader:
            v_d, v_mask, v_p, y = v_d.to(device), v_mask.to(device), v_p.to(device), y.to(device)
            output = model(v_d, v_mask, v_p)
            val_loss += criterion(output, y).item()
            
            probs = torch.sigmoid(output).cpu().numpy()
            y_pred.extend(probs)
            y_true.extend(y.cpu().numpy())
            
    val_loss /= len(loader)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    auc = roc_auc_score(y_true, y_pred)
    acc = accuracy_score(y_true, (y_pred > 0.5).astype(int))
    f1 = f1_score(y_true, (y_pred > 0.5).astype(int))
    
    return val_loss, auc, acc, f1

def main():
    parser = argparse.ArgumentParser(description='DeepTTC Benchmarking Training')
    parser.add_argument('--data_dir', type=str, default='../benchmark formatted dataset', help='Path to formatted dataset')
    parser.add_argument('--vocab_dir', type=str, default='..', help='Path to DeepTTC-main root (for ESPF)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--model_save_path', type=str, default='model_checkpoint.pt', help='Where to save the model')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Data
    print("Loading data...")
    res_df, exp_df, drug_df = load_formatted_data(args.data_dir)
    
    # Split (Simple 80/20 for this example script)
    train_size = int(0.8 * len(res_df))
    train_res = res_df.iloc[:train_size]
    val_res = res_df.iloc[train_size:]

    encoder = BPE_Encoder(args.vocab_dir)
    
    train_dataset = DeepTTC_Dataset(train_res, exp_df, drug_df, encoder)
    val_dataset = DeepTTC_Dataset(val_res, exp_df, drug_df, encoder)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize Model
    model = DeepTTC_Model().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0
    for epoch in range(1, args.epochs + 1):
        loss = train(model, device, train_loader, optimizer, criterion, epoch)
        v_loss, auc, acc, f1 = evaluate(model, device, val_loader, criterion)
        
        print(f"Epoch {epoch}: Val Loss: {v_loss:.4f} | AUC: {auc:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), args.model_save_path)
            print(f"--- Model saved with AUC: {best_auc:.4f} ---")

if __name__ == "__main__":
    main()
