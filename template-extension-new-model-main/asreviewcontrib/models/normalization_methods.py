import numpy as np
import cupy as cp
import torch
from scipy.sparse import csr_matrix, isspmatrix_csr
from cupyx.scipy.special import erf

def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print("Using GPU")
    else:
        print("Using CPU")
    return device

def to_gpu(tensor):
    print(" TENSOR TO GPU")
    return cp.array(tensor) if isinstance(tensor, np.ndarray) else tensor

def to_cpu(tensor):
    print(" TENSOR TO CPU")

    return cp.asnumpy(tensor) if isinstance(tensor, cp.ndarray) else np.array(tensor)

def minmax_fn(tensor, new_min=0, new_max=1):
    print("########## MIN-MAX function #########")
    min_val = cp.min(tensor) if isinstance(tensor, cp.ndarray) else np.min(tensor)
    max_val = cp.max(tensor) if isinstance(tensor, cp.ndarray) else np.max(tensor)
    return ((tensor - min_val) / (max_val - min_val)) * (new_max - new_min) + new_min if max_val != min_val else tensor

def absmin_fn(tensor):
    print("############## absmin function #############")
    
    min_val = cp.min(tensor) if isinstance(tensor, cp.ndarray) else np.min(tensor)
    abs_min_val = cp.abs(min_val) if isinstance(tensor, cp.ndarray) else np.abs(min_val)
    return tensor + abs_min_val if min_val < 0 else tensor

def sqrt_fn(tensor):
    print("############## sqrt function #############")
    return cp.sqrt(cp.abs(tensor)) if isinstance(tensor, cp.ndarray) else np.sqrt(np.abs(tensor))

def cdf_fn(tensor):
    print("############## CDF function #############")
    mean = cp.mean(tensor) if isinstance(tensor, cp.ndarray) else np.mean(tensor)
    std = cp.std(tensor) if isinstance(tensor, cp.ndarray) else np.std(tensor)
    standardized_tensor = (tensor - mean) / std
    return 0.5 * (1 + erf(standardized_tensor / cp.sqrt(2))) if isinstance(tensor, cp.ndarray) else 0.5 * (1 + np.erf(standardized_tensor / np.sqrt(2)))

def sigmoid_fn(tensor):
    print("############## sigmoid function #############")
    return cp.divide(1, (1 + cp.exp(-tensor))) if isinstance(tensor, cp.ndarray) else 1 / (1 + np.exp(-tensor))

def zscore_fn(tensor):
    print("############## zscore function #############")
    if isinstance(tensor, cp.ndarray):
        mean = cp.mean(tensor)
        std = cp.std(tensor)
        return (tensor - mean) / std if std != 0 else cp.zeros_like(tensor)
    else:
        mean = np.mean(tensor)
        std = np.std(tensor)
        return (tensor - mean) / std if std != 0 else np.zeros_like(tensor)

def pareto_fn(tensor):
    print("############## pareto function #############")
    if isinstance(tensor, cp.ndarray):
        mean = cp.mean(tensor)
        std = cp.sqrt(cp.std(tensor))
        return (tensor - mean) / std if std != 0 else cp.zeros_like(tensor)
    else:
        mean = np.mean(tensor)
        std = np.sqrt(np.std(tensor))
        return (tensor - mean) / std if std != 0 else np.zeros_like(tensor)

def l2_normalize_fn(tensor):
    print("############## l2 normalize function #############")
    if isinstance(tensor, cp.ndarray):
        l2_norm = cp.linalg.norm(tensor)
        return tensor / l2_norm if l2_norm != 0 else tensor
    else:
        l2_norm = np.linalg.norm(tensor)
        return tensor / l2_norm if l2_norm != 0 else tensor

def apply_transformations(vector, transformations):
    device = get_device()
    use_gpu = device.type == 'cuda'
    print("############## Using Cuda:" , device.type == 'cuda', " #############")
    
    
    if isspmatrix_csr(vector):
        vector = vector.toarray()  # Convert sparse matrix to dense array for processing
    
    tensor = to_gpu(vector) if use_gpu else np.array(vector)

    for transform_fn in transformations:
        print(f"Applying {transform_fn.__name__} on {device}")
        tensor = transform_fn(tensor)

    result = to_cpu(tensor) if use_gpu else np.array(tensor)
    return csr_matrix(result) if isspmatrix_csr(vector) else result

# Define scaling functions
scaling_functions = {
    'minmax': minmax_fn,
    'absmin': absmin_fn,
    'sqrt': sqrt_fn,
    'cdf': cdf_fn,
    'sigmoid': sigmoid_fn
}

# Define normalization functions
normalization_functions = {
    'zscore': zscore_fn,
    'pareto': pareto_fn,
    'l2_normalize': l2_normalize_fn
}

# Generate individual scaling functions dynamically
for scale_name, scale_fn in scaling_functions.items():
    globals()[scale_name] = lambda vector, scale_fn=scale_fn: apply_transformations(vector, [scale_fn])

# Generate combined normalization and scaling functions dynamically
for norm_name, norm_fn in normalization_functions.items():
    for scale_name, scale_fn in scaling_functions.items():
        func_name = f"{norm_name}_{scale_name}"
        globals()[func_name] = lambda vector, norm_fn=norm_fn, scale_fn=scale_fn: apply_transformations(vector, [norm_fn, scale_fn])

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.sparse
from sklearn.preprocessing import normalize

def add_embeddings(X, name):
    # print(f"################# Adding to embedding file and creating plots for {name} ################")
    
    # # Convert sparse matrix to dense if necessary
    # if scipy.sparse.issparse(X):
    #     print(f"Converting sparse matrix of shape {X.shape} to dense")
    #     X = X.toarray()
    
    # print(f"Shape of dense X: {X.shape}")
    
    # # Check if X is a 2D array
    # if len(X.shape) != 2:
    #     print(f"Error: Expected a 2D array, but got an array with shape {X.shape}")
    #     return

    # # File handling
    # csv_file_path = r"C:\Users\Sjard\OneDrive - Universiteit Utrecht\Thesis\ASReview\Simulation_study_PTSD\Makita\fe_em\Graphs"
    # if os.path.exists(csv_file_path):
    #     os.remove(csv_file_path)
    #     print(f"Deleted existing file: {csv_file_path}")

    # # # Create new DataFrame and save to CSV
    # # new_data = pd.DataFrame(X, columns=[f"dim_{i+1}" for i in range(X.shape[1])])
    # # new_data.insert(0, 'name', [f"{name}_{i}" for i in range(X.shape[0])])
    # # new_data.to_csv(csv_file_path, index=False)
    # # print(f"Embeddings saved to {csv_file_path}")

    # # Create plots
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # # Histogram plot
    # flattened_X = X.flatten()
    # ax1.hist(flattened_X, bins=100, alpha=0.75, edgecolor='black')
    # ax1.set_title(f"Histogram of {name} Values")
    # ax1.set_xlabel("Value")
    # ax1.set_ylabel("Frequency")
    # ax1.set_yscale('log')
    # ax1.grid(True)

    # # PCA plot
    # pca = PCA(n_components=2)
    # pca_result = pca.fit_transform(X)
    
    # # Calculate vector magnitudes for coloring
    # magnitudes = np.linalg.norm(X, axis=1)
    
    # scatter = ax2.scatter(pca_result[:, 0], pca_result[:, 1], 
    #                       c=magnitudes, cmap='viridis', 
    #                       alpha=0.5, s=1)
    # ax2.set_title(f"PCA of {name} Document Embeddings")
    # ax2.set_xlabel("Principal Component 1")
    # ax2.set_ylabel("Principal Component 2")
    # ax2.grid(True)
    # plt.colorbar(scatter, ax=ax2, label='Vector Magnitude')

    # plt.tight_layout()
    
    # # Save the plot
    # plot_file_path = os.path.join(os.path.dirname(csv_file_path), f"{name}_plots.png")
    # if os.path.exists(plot_file_path):
    #     os.remove(plot_file_path)
    #     print(f"Deleted existing plot: {plot_file_path}")
    # plt.savefig(plot_file_path, dpi=300, bbox_inches='tight')
    # plt.close(fig)
    # print(f"Plots saved to {plot_file_path}")
    print("Method fired without execution.")