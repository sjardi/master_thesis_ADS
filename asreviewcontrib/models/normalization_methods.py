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
    return cp.array(tensor) if isinstance(tensor, np.ndarray) else tensor

def to_cpu(tensor):
    return cp.asnumpy(tensor) if isinstance(tensor, cp.ndarray) else np.array(tensor)

def minmax_fn(tensor, new_min=0, new_max=1):
    print("##########%#%#%##% " , isinstance(tensor, cp.ndarray), "#%%%%%%%%%%%%%%$$$$$$$")
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

# Error :  raise AttributeError(f"module 'cupy' has no attribute {name!r}") / AttributeError: module 'cupy' has no attribute 'erf'
def cdf_fn(tensor):
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


# import cupy as cp
# import numpy as np
# from scipy.sparse import csr_matrix, isspmatrix_csr
# from scipy.stats import norm
# import torch


# def get_device():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     if device.type == 'cuda':
#         print("Using GPU")
#     else:
#         print("Using CPU")
#     return device


# def minmax(x, new_min=0, new_max=1):
#     """Normalize the input data to a new range [new_min, new_max].

#     Parameters:
#     - x: Input data, can be a sparse matrix (csr_matrix) or a dense array.
#     - new_min: New minimum value after normalization.
#     - new_max: New maximum value after normalization.

#     Returns:
#     - Normalized data with values scaled to the range [new_min, new_max].
#     """
#     device = get_device()
#     if isinstance(x, csr_matrix):
#         x_dense = x.toarray()
#         x_tensor = torch.tensor(x_dense, device=device)
#         x_min = torch.min(x_tensor)
#         x_max = torch.max(x_tensor)
#         x_scaled = ((x_tensor - x_min) / (x_max - x_min)) * \
#             (new_max - new_min) + new_min
#         return x_scaled.cpu().numpy()
#     else:
#         x_tensor = torch.tensor(x, device=device)
#         x_min = torch.min(x_tensor)
#         x_max = torch.max(x_tensor)
#         x_scaled = ((x_tensor - x_min) / (x_max - x_min)) * \
#             (new_max - new_min) + new_min
#         return x_scaled.cpu().numpy()


# def absmin(x):
#     """Shift input data to ensure all values are non-negative by adding the absolute value of the minimum value.

#     Parameters:
#     - x: Input data, can be a sparse matrix (csr_matrix) or a dense array.

#     Returns:
#     - Data shifted to be non-negative.
#     """
#     device = get_device()
#     if isinstance(x, csr_matrix):
#         x_dense = x.toarray()
#         x_tensor = torch.tensor(x_dense, device=device)
#         min_value = torch.min(x_tensor)
#         if min_value < 0:
#             x_shifted = x_tensor + torch.abs(min_value)
#             return x_shifted.cpu().numpy()
#         return x_dense
#     else:
#         x_tensor = torch.tensor(x, device=device)
#         min_value = torch.min(x_tensor)
#         if min_value < 0:
#             x_shifted = x_tensor + torch.abs(min_value)
#             return x_shifted.cpu().numpy()
#         return x


def relu(x):
    """Apply the ReLU function to input data.

    Parameters:
    - x: Input data, can be a sparse matrix (csr_matrix) or a dense array.

    Returns:
    - Data with ReLU applied.
    """
    device = get_device()
    if isinstance(x, csr_matrix):
        x_dense = x.toarray()
        x_tensor = torch.tensor(x_dense, device=device)
        x_relu = torch.relu(x_tensor)
        return x_relu.cpu().numpy()
    else:
        x_tensor = torch.tensor(x, device=device)
        x_relu = torch.relu(x_tensor)
        return x_relu.cpu().numpy()


# def softplus(x):
#     """Apply the Softplus function to input data.

#     Parameters:
#     - x: Input data, can be a sparse matrix (csr_matrix) or a dense array.

#     Returns:
#     - Data with Softplus applied.
#     """
#     device = get_device()
#     if isinstance(x, csr_matrix):
#         x_dense = x.toarray()
#         x_tensor = torch.tensor(x_dense, device=device)
#         x_softplus = torch.log1p(torch.exp(x_tensor))
#         return x_softplus.cpu().numpy()
#     else:
#         x_tensor = torch.tensor(x, device=device)
#         x_softplus = torch.log1p(torch.exp(x_tensor))
#         return x_softplus.cpu().numpy()


# def sigmoid(x):
#     """Apply the Sigmoid function to input data.

#     Parameters:
#     - x: Input data, can be a sparse matrix (csr_matrix) or a dense array.

#     Returns:
#     - Data with Sigmoid applied.
#     """
#     device = get_device()
#     if isinstance(x, csr_matrix):
#         x_dense = x.toarray()
#         x_tensor = torch.tensor(x_dense, device=device)
#         x_sigmoid = torch.sigmoid(x_tensor)
#         return x_sigmoid.cpu().numpy()
#     else:
#         x_tensor = torch.tensor(x, device=device)
#         x_sigmoid = torch.sigmoid(x_tensor)
#         return x_sigmoid.cpu().numpy()


# def cdf(x):
#     """Apply the cumulative distribution function (CDF) to input data.

#     Parameters:
#     - x: Input data, can be a sparse matrix (csr_matrix) or a dense array.

#     Returns:
#     - Data with CDF applied.
#     """
#     device = get_device()
#     if isinstance(x, csr_matrix):
#         x_dense = x.toarray()
#         x_tensor = torch.tensor(x_dense, device=device)
#         mean = torch.mean(x_tensor)
#         std = torch.std(x_tensor)
#         standardized_x = (x_tensor - mean) / std
#         x_cdf = 0.5 * (1 + torch.erf(standardized_x /
#                        torch.sqrt(torch.tensor(2.0, device=device))))
#         return x_cdf.cpu().numpy()
#     else:
#         x_tensor = torch.tensor(x, device=device)
#         mean = torch.mean(x_tensor)
#         std = torch.std(x_tensor)
#         standardized_x = (x_tensor - mean) / std
#         x_cdf = 0.5 * (1 + torch.erf(standardized_x /
#                        torch.sqrt(torch.tensor(2.0, device=device))))
#         return x_cdf.cpu().numpy()


# def l2_normalize(vector):
#     if isspmatrix_csr(vector):
#         # Compute the L2 norm of the CSR matrix
#         l2_norm = np.sqrt(vector.multiply(vector).sum())

#         # Handle the case where the norm is zero to avoid division by zero
#         if l2_norm == 0:
#             return vector

#         # Normalize the CSR matrix
#         normalized_vector = vector / l2_norm
#     elif isinstance(vector, np.ndarray):
#         # Compute the L2 norm of the NumPy array
#         l2_norm = np.linalg.norm(vector)

#         # Handle the case where the norm is zero to avoid division by zero
#         if l2_norm == 0:
#             return vector

#         # Normalize the NumPy array
#         normalized_vector = vector / l2_norm
#     else:
#         raise ValueError("Input must be a NumPy array or a CSR matrix")
#     normalized_vector = l2_normalize(vector)
#     return normalized_vector.cpu().numpy()


# def sqrt(vector):
#     if isinstance(vector, np.ndarray):
#         # Convert the NumPy array to a CuPy array
#         vector_gpu = cp.asarray(vector)
#         # Apply square root on the GPU
#         sqrt_vector_gpu = cp.sqrt(cp.abs(vector_gpu))
#         # Convert back to a NumPy array
#         sqrt_vector = cp.asnumpy(sqrt_vector_gpu)
#     elif isspmatrix_csr(vector):
#         # Convert the CSR matrix to a CuPy sparse matrix
#         vector_gpu = cp.sparse.csr_matrix(vector)
#         # Apply square root on the GPU
#         sqrt_vector_gpu = cp.sqrt(cp.abs(vector_gpu))
#         # Convert back to a SciPy sparse CSR matrix
#         sqrt_vector = cp.sparse.csr_matrix(sqrt_vector_gpu)
#     else:
#         raise ValueError("Input must be a NumPy array or a CSR matrix")

#     return sqrt_vector

# ###############################################

# def sqrt_gpu(vector_gpu):
#     print("Applying square root scaling on GPU")
#     return cp.sqrt(cp.abs(vector_gpu))


# def zscore_gpu(vector_gpu):
#     print("Applying Z-score normalization on GPU")
#     mean = cp.mean(vector_gpu)
#     std = cp.std(vector_gpu)

#     if std == 0:
#         return vector_gpu

#     return (vector_gpu - mean) / std


# def sqrt_cpu(vector):
#     print("Applying square root scaling on CPU")
#     return np.sqrt(np.abs(vector))


# def zscore_cpu(vector):
#     print("Applying Z-score normalization on CPU")
#     mean = np.mean(vector)
#     std = np.std(vector)

#     if std == 0:
#         return vector

#     return (vector - mean) / std

# def zscore_squareroot(vector):
#     use_gpu = cp.is_available()

#     if use_gpu:
#         if isinstance(vector, np.ndarray):
#             vector_gpu = cp.asarray(vector)
#         elif isspmatrix_csr(vector):
#             vector_gpu = cp.sparse.csr_matrix(vector)
#         else:
#             raise ValueError("Input must be a NumPy array or a CSR matrix")

#         vector_gpu = zscore_gpu(vector_gpu)
#         vector_gpu = sqrt_gpu(vector_gpu)

#         if isinstance(vector, np.ndarray):
#             return cp.asnumpy(vector_gpu)
#         elif isspmatrix_csr(vector):
#             return cp.sparse.csr_matrix(vector_gpu)
#     else:
#         if isinstance(vector, np.ndarray):
#             vector = zscore_cpu(vector)
#             vector = sqrt_cpu(vector)
#             return vector
#         elif isspmatrix_csr(vector):
#             vector = vector.toarray()
#             vector = zscore_cpu(vector)
#             vector = sqrt_cpu(vector)
#             return csr_matrix(vector)
#         else:
#             raise ValueError("Input must be a NumPy array or a CSR matrix")


# # import numpy as np
# # from scipy.sparse import csr_matrix
# # from scipy.stats import norm

# # def minmax(x, new_min=0, new_max=1):
# #     """Normalize the input data to a new range [new_min, new_max].

# #     Parameters:
# #     - x: Input data, can be a sparse matrix (csr_matrix) or a dense array.
# #     - new_min: New minimum value after normalization.
# #     - new_max: New maximum value after normalization.

# #     Returns:
# #     - Normalized data with values scaled to the range [new_min, new_max].
# #     """
# #     if isinstance(x, csr_matrix):
# #         x_min = x.data.min()
# #         x_max = x.data.max()
# #         x.data = ((x.data - x_min) / (x_max - x_min)) * (new_max - new_min) + new_min
# #         return x
# #     else:
# #         x_min = np.min(x)
# #         x_max = np.max(x)
# #         return ((x - x_min) / (x_max - x_min)) * (new_max - new_min) + new_min

# # def absmin(x):
# #     """Shift input data to ensure all values are non-negative by adding the absolute value of the minimum value.

# #     Parameters:
# #     - x: Input data, can be a sparse matrix (csr_matrix) or a dense array.

# #     Returns:
# #     - Data shifted to be non-negative.
# #     """
# #     if isinstance(x, csr_matrix):
# #         min_value = x.data.min()
# #         if (min_value < 0):
# #             x.data += np.abs(min_value)
# #         return x
# #     else:
# #         min_value = np.min(x)
# #         if (min_value < 0):
# #             x += np.abs(min_value)
# #         return x

# # def relu(x):
# #     """Apply the ReLU function to input data.

# #     Parameters:
# #     - x: Input data, can be a sparse matrix (csr_matrix) or a dense array.

# #     Returns:
# #     - Data with ReLU applied.
# #     """
# #     if isinstance(x, csr_matrix):
# #         x.data = np.maximum(0, x.data)
# #         return x
# #     else:
# #         return np.maximum(0, x)

# # def softplus(x):
# #     """Apply the Softplus function to input data.

# #     Parameters:
# #     - x: Input data, can be a sparse matrix (csr_matrix) or a dense array.

# #     Returns:
# #     - Data with Softplus applied.
# #     """
# #     if isinstance(x, csr_matrix):
# #         x.data = np.log1p(np.exp(x.data))
# #         return x
# #     else:
# #         return np.log1p(np.exp(x))

# # def sigmoid(x):
# #     """Apply the Sigmoid function to input data.

# #     Parameters:
# #     - x: Input data, can be a sparse matrix (csr_matrix) or a dense array.

# #     Returns:
# #     - Data with Sigmoid applied.
# #     """
# #     if isinstance(x, csr_matrix):
# #         x.data = 1 / (1 + np.exp(-x.data))
# #         return x
# #     else:
# #         return 1 / (1 + np.exp(-x))

# # def cdf(x):
# #     """Apply the cumulative distribution function (CDF) to input data.

# #     Parameters:
# #     - x: Input data, can be a sparse matrix (csr_matrix) or a dense array.

# #     Returns:
# #     - Data with CDF applied.
# #     """
# #     if isinstance(x, csr_matrix):
# #         # Standardize data
# #         mean = np.mean(x.data)
# #         std = np.std(x.data)
# #         x.data = (x.data - mean) / std
# #         # Apply CDF to data
# #         x.data = norm.cdf(x.data)
# #         return x
# #     else:
# #         # Standardize data
# #         mean = np.mean(x)
# #         std = np.std(x)
# #         standardized_x = (x - mean) / std
# #         # Apply CDF to data
# #         return norm.cdf(standardized_x)


# # # normalization_methods.py
# # import numpy as np
# # from scipy.sparse import csr_matrix
# # from scipy.stats import norm

# # def minmax(x, new_min=0, new_max=1):
# #     """Normalize the input data to a new range [new_min, new_max].

# #     Parameters:
# #     - x: Input data, can be a sparse matrix (csr_matrix) or a dense array.
# #     - new_min: New minimum value after normalization.
# #     - new_max: New maximum value after normalization.

# #     Returns:
# #     - Normalized data with values scaled to the range [new_min, new_max].
# #     """
# #     if isinstance(x, csr_matrix):
# #         x_min = x.data.min()
# #         x_max = x.data.max()
# #         x.data = ((x.data - x_min) / (x_max - x_min)) * (new_max - new_min) + new_min
# #         return x
# #     else:
# #         x_min = np.min(x)
# #         x_max = np.max(x)
# #         return ((x - x_min) / (x_max - x_min)) * (new_max - new_min) + new_min

# # def absmin(x):
# #     """Shift input data to ensure all values are non-negative by adding the absolute value of the minimum value.

# #     Parameters:
# #     - x: Input data, can be a sparse matrix (csr_matrix) or a dense array.

# #     Returns:
# #     - Data shifted to be non-negative.
# #     """
# #     if isinstance(x, csr_matrix):
# #         min_value = x.data.min()
# #         if min_value < 0:
# #             x.data += np.abs(min_value)
# #         return x
# #     else:
# #         min_value = np.min(x)
# #         if min_value < 0:
# #             x += np.abs(min_value)
# #         return x

# # def relu(x):
# #     # Check if the input is a sparse matrix in CSR format
# #     if isinstance(x, csr_matrix):
# #         # Apply ReLU to the data array of the sparse matrix
# #         x.data = np.maximum(0, x.data)
# #         return x
# #     else:
# #         # Fallback for dense arrays
# #         return np.maximum(0, x)

# # def softplus(x):
# #         return np.log1p(np.exp(x))

# # def sigmoid(x):
# #     return 1 / (1 + np.exp(-x.data))

# # def cdf(x):
# #     x = norm.cdf(x.toarray())
# #     return csr_matrix(x)
