import math
import torch
import numpy as np
from sklearn.metrics import roc_auc_score,precision_recall_curve,accuracy_score
def uniform(size, tensor):
    if tensor is not None:
        bound = 1.0 / math.sqrt(size)
        tensor.data.uniform_(-bound, bound)
def kaiming_uniform(tensor, fan, a):
    if tensor is not None:
        bound = math.sqrt(6 / ((1 + a**2) * fan))
        tensor.data.uniform_(-bound, bound)
def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)
def glorot_orthogonal(tensor, scale):
    if tensor is not None:
        torch.nn.init.orthogonal_(tensor.data)
        scale /= ((tensor.size(-2) + tensor.size(-1)) * tensor.var())
        tensor.data *= scale.sqrt()
def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)
def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(1)
def normal(tensor, mean, std):
    if tensor is not None:
        tensor.data.normal_(mean, std)
def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()
    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

def metrics_graph(yt, yp):
    precision, recall, _, = precision_recall_curve(yt, yp)
    aupr = -np.trapz(precision, recall)
    auc = roc_auc_score(yt, yp)
    #---f1,acc,recall, specificity, precision
    # Convert to 2D arrays (row vectors) to match np.mat() behavior
    # np.mat() converts 1D arrays to row vectors (1, n)
    real_score = np.atleast_2d(np.asarray(yt).flatten())  # Row vector (1, n)
    predict_score = np.atleast_2d(np.asarray(yp).flatten())  # Row vector (1, n)
    sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(sorted_predict_score_num * np.arange(1, 1000) / 1000)]
    thresholds = np.atleast_2d(np.asarray(thresholds).flatten())  # Row vector (1, n)
    thresholds_num = thresholds.shape[1]
    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)  # (thresholds_num, 1)
    FP = predict_score_matrix.sum(axis=1, keepdims=True) - TP  # (thresholds_num, 1)
    FN = real_score.sum() - TP  # scalar - (thresholds_num, 1) = (thresholds_num, 1)
    total_samples = real_score.shape[1]  # Number of samples (n)
    TN = total_samples - TP - FP - FN  # (thresholds_num, 1)
    
    # Avoid division by zero
    tpr = np.divide(TP, TP + FN, out=np.zeros_like(TP), where=(TP + FN) != 0)  # recall
    recall_list = tpr
    precision_list = np.divide(TP, TP + FP, out=np.zeros_like(TP), where=(TP + FP) != 0)
    
    # F1 = 2 * (precision * recall) / (precision + recall)
    # Alternative: F1 = 2 * TP / (2 * TP + FP + FN)
    f1_score_list = np.divide(2 * TP, 2 * TP + FP + FN, out=np.zeros_like(TP), where=(2 * TP + FP + FN) != 0)
    
    # Accuracy = (TP + TN) / total_samples
    accuracy_list = (TP + TN) / total_samples
    
    specificity_list = np.divide(TN, TN + FP, out=np.zeros_like(TN), where=(TN + FP) != 0)
    
    max_index = np.argmax(f1_score_list)
    # Extract scalar values - f1_score_list is (thresholds_num, 1), np.argmax returns flattened index
    # Use .flat[] to extract scalar from 2D array using flattened index
    f1_score = float(f1_score_list.flat[max_index])
    accuracy = float(accuracy_list.flat[max_index])
    specificity = float(specificity_list.flat[max_index])
    recall = float(recall_list.flat[max_index])
    precision = float(precision_list.flat[max_index])
    
    # Ensure metrics are in valid range [0, 1]
    f1_score = np.clip(f1_score, 0.0, 1.0)
    accuracy = np.clip(accuracy, 0.0, 1.0)
    return auc, aupr, f1_score, accuracy #, recall, specificity, precision