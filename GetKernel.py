import numpy as np
from scipy import sparse

def get_kernel(features_a, features_b=None, param=None):
    """
    Compute a kernel, it can be K(A, A) or K(A, B)
    
    Usage:
        1. Compute the kernel between different examples, e.g., in testing:
            kernel, param = get_kernel(features_a, features_b, param)
        2. Compute the kernel between the same examples, e.g., in training:
            kernel, param = get_kernel(features, param)
    
    Input:
        features_a: d-by-m numpy array, d is feature dimension, m is number of samples
        features_b: d-by-n numpy array, d is feature dimension, n is number of samples
        param: dictionary with keys:
            - 'kernel_type': 'linear' or 'gaussian'
            - For gaussian: 'ratio', 'sigma', or 'gamma' (depending on implementation)
    
    Output:
        kernel: m-by-n or m-by-m numpy array
        param: updated parameter dictionary
    """
    
    # Handle input arguments
    if features_b is None and param is None:
        raise ValueError("Not enough inputs!")
    elif param is None:
        param = features_b
        features_b = features_a
    
    # Check kernel type
    if 'kernel_type' not in param:
        raise ValueError("Please specify the kernel_type!")
    
    kernel_type = param['kernel_type'].lower()
    
    if kernel_type == 'linear':
        kernel = _linear_kernel(features_a, features_b)
    elif kernel_type == 'gaussian':
        kernel, param = _gaussian_kernel(features_a, features_b, param)
    else:
        raise ValueError(f"Unknown type of kernel: {param['kernel_type']}")
    
    return kernel, param

def _linear_kernel(features_a, features_b, param=None):
    """
    Compute linear kernel: K(x,y) = x^T y
    
    Parameters:
    -----------
    features_a : array-like, shape (d, nA)
        First set of features (d dimensions, nA samples)
    features_b : array-like, shape (d, nB)
        Second set of features (d dimensions, nB samples)
    param : dict, optional
        Additional parameters (not used in linear kernel but kept for consistency)
    
    Returns:
    --------
    K : ndarray, shape (nA, nB)
        Computed kernel matrix
    param : dict
        Unmodified input parameters
    """
    
    # Convert inputs to numpy arrays if they aren't already
    features_a = np.asarray(features_a)
    features_b = np.asarray(features_b)
    
    # Check dimensions match
    dA, nA = features_a.shape
    dB, nB = features_b.shape
    
    if dA != dB:
        raise ValueError(f"Feature dimensions don't match: {dA} != {dB}")
    
    # Compute linear kernel
    K = features_a.T @ features_b
    
    # Convert sparse matrix to dense if needed (similar to MATLAB's full())
    if sparse.issparse(K):
        K = K.toarray()
    
    return K, param

def _gaussian_kernel(features_a, features_b, param):
    """
    Compute Gaussian (RBF) kernel: K(x,y) = exp(-gamma * ||x-y||^2)
    
    Parameters:
    -----------
    features_a : ndarray, shape (d, nA)
        First set of features (d dimensions, nA samples)
    features_b : ndarray, shape (d, nB)
        Second set of features (d dimensions, nB samples)
    param : dict
        Dictionary containing kernel parameters:
        - 'ratio': scaling ratio (default=1)
        - 'sigma': bandwidth parameter
        - 'gamma': direct kernel parameter
    
    Returns:
    --------
    K : ndarray, shape (nA, nB)
        Computed Gaussian kernel matrix
    param : dict
        Updated parameter dictionary
    """
    
    dA, nA = features_a.shape
    dB, nB = features_b.shape
    
    if dA != dB:
        raise ValueError(f"Feature dimensions don't match: {dA} != {dB}")
    
    # Compute squared Euclidean distance
    sq_dist = l2_distance_squared(features_a, features_b)
    
    # Set default ratio if not specified or 0
    if 'ratio' not in param or param['ratio'] == 0:
        param['ratio'] = 1
    
    # Handle gamma/sigma parameters
    if 'gamma' not in param or param['gamma'] == 0:
        if 'sigma' not in param or param['sigma'] == 0:
            # Use default sigma
            tmp = np.mean(np.mean(sq_dist, axis=0)) * 0.5
            param['sigma'] = np.sqrt(tmp)
            print(tmp)
        
        # Compute gamma based on ratio and sigma
        if param['sigma'] == 0:
            param['gamma'] = 0
        else:
            param['gamma'] = 1 / (2 * param['ratio'] * param['sigma']**2)
    else:
        # Gamma was specified, set sigma and ratio to 0
        if 'sigma' not in param:
            param['sigma'] = 0
        if 'ratio' not in param:
            param['ratio'] = 0
    
    # Compute Gaussian kernel
    K = np.exp(-sq_dist * param['gamma'])
    
    return K, param


def l2_distance_squared(x, c):
    """
    Compute squared L2 distance between two sets of samples (faster implementation)
    
    Parameters:
    -----------
    x : ndarray, shape (d, m)
        First set of vectors (d dimensions, m samples)
    c : ndarray, shape (d, n)
        Second set of vectors (d dimensions, n samples)
    df : int, optional (default=0)
        If df=1, forces diagonal to zero (useful when x == c)
    
    Returns:
    --------
    n2 : ndarray, shape (m, n)
        Squared L2 distance matrix
    """
    
    dimx, ndata = x.shape
    dimc, ncentres = c.shape
    
    if dimx != dimc:
        raise ValueError('Data dimension does not match dimension of centres')
    
    # Compute squared distances using vectorized operations
    sum_x = np.sum(x**2, axis=0, keepdims=True)  # shape (m,)
    sum_c = np.sum(c**2, axis=0, keepdims=True)   # shape (n,)
    
    # Using broadcasting for efficient computation
    n2 = (sum_x.T + sum_c - 2 * np.dot(x.T, c))
    
    # Ensure results are real and non-negative
    n2 = np.real(n2)
    n2[n2 < 0] = 0

    return n2