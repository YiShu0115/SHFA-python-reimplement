import numpy as np
from sklearn.svm import SVC

def LpMKL_HFA(labels, K_root, M, param):
    """
    Python implementation of LpMKL_H_fast for HFA
    
    Parameters:
    - labels: training labels (n_samples,)
    - K_root: square root of kernel matrix (n_samples, n_samples)
    - M: matrix of basis vectors (n_samples, n_basekernels)
    - param: dictionary containing:
        * degree: p in Lp-norm (default 1)
        * d_norm: constraint on d's norm (default 1)
        * weight: sample weights (n_samples,)
        * svm_C: SVM C parameter
        * d: initial coefficients (optional)
    
    Returns:
    - coefficients: learned weights for base kernels
    - model: trained SVM model
    - last_obj: final objective value
    """
    
    n_samples, n_basekernels = M.shape
    
    # Set default parameters
    degree = param.get('degree', 1)
    d_norm = param.get('d_norm', 1)
    # weight = param.get('weight', np.ones(n_samples))
    # svm_C = param.get('svm_C', 1)
    
    # Initialize coefficients
    if 'd' in param:
        d = param['d']
        d = d_norm * d / (np.sum(d**degree)**(1/degree))
        coefficients = np.zeros(n_basekernels)
        coefficients[:len(d)] = d
    else:
        coefficients = d_norm * np.ones(n_basekernels) * (1/n_basekernels)**(1/degree)
    
    # Optimization parameters
    MAX_ITER = 100
    tau = 1e-3
    obj = []
    
    # First iteration
    model, obj_val, wn = return_alpha(K_root, labels, M, coefficients, param)
    obj.append(obj_val)
    
    # Main optimization loop
    for i in range(1, MAX_ITER):
        # Update coefficients
        wnp = wn**(2/(degree + 1))
        eta = (np.sum(wnp**degree))**(1/degree)
        coefficients = d_norm * wnp / eta
        # Solve SVM
        model, obj_val, wn = return_alpha(K_root, labels, M, coefficients, param)
        obj.append(obj_val)

        # Check convergence
        if abs(obj[i] - obj[i-1]) <= tau * abs(obj[i]):
            break
    
    return coefficients, model, obj[-1]

def return_alpha(K_root, labels, M, coefficients, param):
    """
    Helper function to solve SVM and compute objective
    """
    n, m = M.shape
    
    # Compute combined kernel
    kernel = sumkernels(K_root, M, coefficients)
    
    # Train SVM with precomputed kernel
    model = SVC(kernel='precomputed',C=param.get('svm_C', 1), class_weight='balanced') 
    model.fit(kernel, labels, sample_weight=param['weight'])

    # Get support vectors information
    support_vector_indices = model.support_
    alpha = np.zeros(n)
    alpha[support_vector_indices] = np.abs(model.dual_coef_.flatten())
    # # 每行格式：标签 0:序号 1:值1 2:值2 ... n:值n
        # Compute objective
    kay = K_root @ (alpha * labels)
    hkay = M.T @ kay
    q = hkay**2
    obj = np.sum(alpha) - 0.5 * (np.sum(q * coefficients) + kay.T @ kay)
    wn = coefficients * np.sqrt(q)
    return model, obj, wn

def sumkernels(K_root, M, coefficients):
    """
    Combine base kernels using coefficients
    """
    n, m = M.shape
    
    # Scale each column of Hvs by sqrt(coefficient)
    M_scaled = M * np.sqrt(coefficients)
    
    # Compute H matrix
    H = M_scaled @ M_scaled.T
    
    # Compute final kernel
    kernel = K_root @ (H + np.eye(n)) @ K_root
    return kernel
