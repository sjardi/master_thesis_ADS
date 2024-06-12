import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import norm
import torch


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print("Using GPU")
    else:
        print("Using CPU")
    return device


def minmax(x, new_min=0, new_max=1):
    """Normalize the input data to a new range [new_min, new_max].

    Parameters:
    - x: Input data, can be a sparse matrix (csr_matrix) or a dense array.
    - new_min: New minimum value after normalization.
    - new_max: New maximum value after normalization.

    Returns:
    - Normalized data with values scaled to the range [new_min, new_max].
    """
    device = get_device()
    if isinstance(x, csr_matrix):
        x_dense = x.toarray()
        x_tensor = torch.tensor(x_dense, device=device)
        x_min = torch.min(x_tensor)
        x_max = torch.max(x_tensor)
        x_scaled = ((x_tensor - x_min) / (x_max - x_min)) * \
            (new_max - new_min) + new_min
        return x_scaled.cpu().numpy()
    else:
        x_tensor = torch.tensor(x, device=device)
        x_min = torch.min(x_tensor)
        x_max = torch.max(x_tensor)
        x_scaled = ((x_tensor - x_min) / (x_max - x_min)) * \
            (new_max - new_min) + new_min
        return x_scaled.cpu().numpy()


def absmin(x):
    """Shift input data to ensure all values are non-negative by adding the absolute value of the minimum value.

    Parameters:
    - x: Input data, can be a sparse matrix (csr_matrix) or a dense array.

    Returns:
    - Data shifted to be non-negative.
    """
    device = get_device()
    if isinstance(x, csr_matrix):
        x_dense = x.toarray()
        x_tensor = torch.tensor(x_dense, device=device)
        min_value = torch.min(x_tensor)
        if min_value < 0:
            x_shifted = x_tensor + torch.abs(min_value)
            return x_shifted.cpu().numpy()
        return x_dense
    else:
        x_tensor = torch.tensor(x, device=device)
        min_value = torch.min(x_tensor)
        if min_value < 0:
            x_shifted = x_tensor + torch.abs(min_value)
            return x_shifted.cpu().numpy()
        return x


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


def softplus(x):
    """Apply the Softplus function to input data.

    Parameters:
    - x: Input data, can be a sparse matrix (csr_matrix) or a dense array.

    Returns:
    - Data with Softplus applied.
    """
    device = get_device()
    if isinstance(x, csr_matrix):
        x_dense = x.toarray()
        x_tensor = torch.tensor(x_dense, device=device)
        x_softplus = torch.log1p(torch.exp(x_tensor))
        return x_softplus.cpu().numpy()
    else:
        x_tensor = torch.tensor(x, device=device)
        x_softplus = torch.log1p(torch.exp(x_tensor))
        return x_softplus.cpu().numpy()


def sigmoid(x):
    """Apply the Sigmoid function to input data.

    Parameters:
    - x: Input data, can be a sparse matrix (csr_matrix) or a dense array.

    Returns:
    - Data with Sigmoid applied.
    """
    device = get_device()
    if isinstance(x, csr_matrix):
        x_dense = x.toarray()
        x_tensor = torch.tensor(x_dense, device=device)
        x_sigmoid = torch.sigmoid(x_tensor)
        return x_sigmoid.cpu().numpy()
    else:
        x_tensor = torch.tensor(x, device=device)
        x_sigmoid = torch.sigmoid(x_tensor)
        return x_sigmoid.cpu().numpy()


def cdf(x):
    """Apply the cumulative distribution function (CDF) to input data.

    Parameters:
    - x: Input data, can be a sparse matrix (csr_matrix) or a dense array.

    Returns:
    - Data with CDF applied.
    """
    device = get_device()
    if isinstance(x, csr_matrix):
        x_dense = x.toarray()
        x_tensor = torch.tensor(x_dense, device=device)
        mean = torch.mean(x_tensor)
        std = torch.std(x_tensor)
        standardized_x = (x_tensor - mean) / std
        x_cdf = 0.5 * (1 + torch.erf(standardized_x /
                       torch.sqrt(torch.tensor(2.0, device=device))))
        return x_cdf.cpu().numpy()
    else:
        x_tensor = torch.tensor(x, device=device)
        mean = torch.mean(x_tensor)
        std = torch.std(x_tensor)
        standardized_x = (x_tensor - mean) / std
        x_cdf = 0.5 * (1 + torch.erf(standardized_x /
                       torch.sqrt(torch.tensor(2.0, device=device))))
        return x_cdf.cpu().numpy()


# import numpy as np
# from scipy.sparse import csr_matrix
# from scipy.stats import norm

# def minmax(x, new_min=0, new_max=1):
#     """Normalize the input data to a new range [new_min, new_max].

#     Parameters:
#     - x: Input data, can be a sparse matrix (csr_matrix) or a dense array.
#     - new_min: New minimum value after normalization.
#     - new_max: New maximum value after normalization.

#     Returns:
#     - Normalized data with values scaled to the range [new_min, new_max].
#     """
#     if isinstance(x, csr_matrix):
#         x_min = x.data.min()
#         x_max = x.data.max()
#         x.data = ((x.data - x_min) / (x_max - x_min)) * (new_max - new_min) + new_min
#         return x
#     else:
#         x_min = np.min(x)
#         x_max = np.max(x)
#         return ((x - x_min) / (x_max - x_min)) * (new_max - new_min) + new_min

# def absmin(x):
#     """Shift input data to ensure all values are non-negative by adding the absolute value of the minimum value.

#     Parameters:
#     - x: Input data, can be a sparse matrix (csr_matrix) or a dense array.

#     Returns:
#     - Data shifted to be non-negative.
#     """
#     if isinstance(x, csr_matrix):
#         min_value = x.data.min()
#         if (min_value < 0):
#             x.data += np.abs(min_value)
#         return x
#     else:
#         min_value = np.min(x)
#         if (min_value < 0):
#             x += np.abs(min_value)
#         return x

# def relu(x):
#     """Apply the ReLU function to input data.

#     Parameters:
#     - x: Input data, can be a sparse matrix (csr_matrix) or a dense array.

#     Returns:
#     - Data with ReLU applied.
#     """
#     if isinstance(x, csr_matrix):
#         x.data = np.maximum(0, x.data)
#         return x
#     else:
#         return np.maximum(0, x)

# def softplus(x):
#     """Apply the Softplus function to input data.

#     Parameters:
#     - x: Input data, can be a sparse matrix (csr_matrix) or a dense array.

#     Returns:
#     - Data with Softplus applied.
#     """
#     if isinstance(x, csr_matrix):
#         x.data = np.log1p(np.exp(x.data))
#         return x
#     else:
#         return np.log1p(np.exp(x))

# def sigmoid(x):
#     """Apply the Sigmoid function to input data.

#     Parameters:
#     - x: Input data, can be a sparse matrix (csr_matrix) or a dense array.

#     Returns:
#     - Data with Sigmoid applied.
#     """
#     if isinstance(x, csr_matrix):
#         x.data = 1 / (1 + np.exp(-x.data))
#         return x
#     else:
#         return 1 / (1 + np.exp(-x))

# def cdf(x):
#     """Apply the cumulative distribution function (CDF) to input data.

#     Parameters:
#     - x: Input data, can be a sparse matrix (csr_matrix) or a dense array.

#     Returns:
#     - Data with CDF applied.
#     """
#     if isinstance(x, csr_matrix):
#         # Standardize data
#         mean = np.mean(x.data)
#         std = np.std(x.data)
#         x.data = (x.data - mean) / std
#         # Apply CDF to data
#         x.data = norm.cdf(x.data)
#         return x
#     else:
#         # Standardize data
#         mean = np.mean(x)
#         std = np.std(x)
#         standardized_x = (x - mean) / std
#         # Apply CDF to data
#         return norm.cdf(standardized_x)


# # normalization_methods.py
# import numpy as np
# from scipy.sparse import csr_matrix
# from scipy.stats import norm

# def minmax(x, new_min=0, new_max=1):
#     """Normalize the input data to a new range [new_min, new_max].

#     Parameters:
#     - x: Input data, can be a sparse matrix (csr_matrix) or a dense array.
#     - new_min: New minimum value after normalization.
#     - new_max: New maximum value after normalization.

#     Returns:
#     - Normalized data with values scaled to the range [new_min, new_max].
#     """
#     if isinstance(x, csr_matrix):
#         x_min = x.data.min()
#         x_max = x.data.max()
#         x.data = ((x.data - x_min) / (x_max - x_min)) * (new_max - new_min) + new_min
#         return x
#     else:
#         x_min = np.min(x)
#         x_max = np.max(x)
#         return ((x - x_min) / (x_max - x_min)) * (new_max - new_min) + new_min

# def absmin(x):
#     """Shift input data to ensure all values are non-negative by adding the absolute value of the minimum value.

#     Parameters:
#     - x: Input data, can be a sparse matrix (csr_matrix) or a dense array.

#     Returns:
#     - Data shifted to be non-negative.
#     """
#     if isinstance(x, csr_matrix):
#         min_value = x.data.min()
#         if min_value < 0:
#             x.data += np.abs(min_value)
#         return x
#     else:
#         min_value = np.min(x)
#         if min_value < 0:
#             x += np.abs(min_value)
#         return x

# def relu(x):
#     # Check if the input is a sparse matrix in CSR format
#     if isinstance(x, csr_matrix):
#         # Apply ReLU to the data array of the sparse matrix
#         x.data = np.maximum(0, x.data)
#         return x
#     else:
#         # Fallback for dense arrays
#         return np.maximum(0, x)

# def softplus(x):
#         return np.log1p(np.exp(x))

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x.data))

# def cdf(x):
#     x = norm.cdf(x.toarray())
#     return csr_matrix(x)
