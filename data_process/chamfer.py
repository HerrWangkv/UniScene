import torch 
import sys 
import os 
from chamferdist.chamfer import ChamferDistance, knn_points 

def forward(x, y): 
    knn_result_xy = knn_points(x, y, K=1) 
    d1 = knn_result_xy.dists.squeeze(-1)  # (B, N) 
    idx1 = knn_result_xy.idx.squeeze(-1)  # (B, N) 
    knn_result_yx = knn_points(y, x, K=1) 
    d2 = knn_result_yx.dists.squeeze(-1)  # (B, M) 
    idx2 = knn_result_yx.idx.squeeze(-1)  # (B, M) 
    return d1, d2, idx1, idx2