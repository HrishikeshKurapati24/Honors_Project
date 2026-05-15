import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from torch_geometric.loader import DataLoader
from model import get_model
from preprocess import GraphDRPDataset
from scipy import stats
import matplotlib.pyplot as plt

# --- Utilities ---

def calculate_metrics(y_true, y_pred):
    mse = ((y_true - y_pred)**2).mean()
    rmse = np.sqrt(mse)
    pearson = np.corrcoef(y_true, y_pred)[0, 1]
    spearman = stats.spearmanr(y_true, y_pred)[0]
    return {"mse": mse, "rmse": rmse, "pearson": pearson, "spearman": spearman}

def train(model, device, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output.view(-1), data.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, device, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            y_true.extend(data.y.cpu().numpy())
            y_pred.extend(output.cpu().numpy())
    return np.array(y_true).flatten(), np.array(y_pred).flatten()

# --- Saliency Analysis ---

def saliency_analysis(model, device, loader, output_file):
    model.eval()
    all_gradients = []
    
    # We must enable gradients for target even in eval mode for saliency
    for data in loader:
        data = data.to(device)
        data.target.requires_grad = True
        output = model(data)
        
        # Loss for saliency is just the output sum (gradient of output w.r.t input)
        output.backward(torch.ones_like(output))
        
        grads = data.target.grad.cpu().detach().numpy()
        all_gradients.append(grads)
    
    avg_grads = np.mean(np.concatenate(all_gradients, axis=0), axis=0)
    np.save(output_file, avg_grads)
    print(f"Saliency map saved to {output_file}")

# --- Main Entry Point ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'saliency'], required=True)
    parser.add_argument('--model_type', type=str, choices=['GCN', 'GAT', 'GIN', 'GAT_GCN'], default='GCN')
    parser.add_argument('--data_dir', type=str, default='processed_pt')
    parser.add_argument('--model_path', type=str, default='best_model.pth')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.0005)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.mode == 'train':
        train_set = GraphDRPDataset(root=args.data_dir, dataset_name='train')
        val_set = GraphDRPDataset(root=args.data_dir, dataset_name='val')
        
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

        model = get_model(args.model_type).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = nn.MSELoss()

        best_mse = float('inf')
        for epoch in range(1, args.epochs + 1):
            loss = train(model, device, train_loader, optimizer, loss_fn)
            y_true, y_pred = evaluate(model, device, val_loader)
            metrics = calculate_metrics(y_true, y_pred)
            
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val MSE: {metrics['mse']:.4f} | Pearson: {metrics['pearson']:.4f}")
            
            if metrics['mse'] < best_mse:
                best_mse = metrics['mse']
                torch.save(model.state_dict(), args.model_path)
                print("--- Best model saved! ---")

    elif args.mode == 'eval':
        test_set = GraphDRPDataset(root=args.data_dir, dataset_name='test')
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
        
        model = get_model(args.model_type).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        
        y_true, y_pred = evaluate(model, device, test_loader)
        metrics = calculate_metrics(y_true, y_pred)
        print("\nFinal Metrics on Test Set:")
        for k, v in metrics.items():
            print(f"{k.upper()}: {v:.4f}")

    elif args.mode == 'saliency':
        dataset = GraphDRPDataset(root=args.data_dir, dataset_name='test', saliency_map=True)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        model = get_model(args.model_type).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        
        saliency_analysis(model, device, loader, "graphdrp_saliency.npy")

if __name__ == "__main__":
    main()
