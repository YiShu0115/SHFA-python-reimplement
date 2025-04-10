import numpy as np
from sklearn.svm import SVC
from MKL_hfa import LpMKL_HFA

def  train_hfa_mkl(slabels, tlabels, K_root, parameters):
    """
    Python implementation of HFA-MKL training
    
    Parameters:
    - slabels: source domain labels (n_s,)
    - tlabels: target domain labels (n_t,)
    - K_root: square root of kernel matrix (n_s+n_t, n_s+n_t)
    - parameters: dictionary containing:
        * C_s: source domain SVM C parameter
        * C_t: target domain SVM C parameter
        * lambda: regularization parameter
        * mkl_degree: p-norm degree (usually 1)
        * hfa_iter: max iterations (default 50)
        * hfa_tau: convergence threshold (default 1e-3)
    
    Returns:
    - model: trained SVM classifier
    - H: learned transformation metric
    - obj: list of objective values
    """
    
    # Set default parameters
    MAX_ITER = parameters.get('hfa_iter', 50)
    tau = parameters.get('hfa_tau', 1e-3)
    
    n_s = len(slabels)
    n_l = len(tlabels)
    n = K_root.shape[0]
    assert n == n_s + n_l, "K_root dimension mismatch"
    
    # Compute weights
    weight = np.concatenate([
        np.ones(n_s) * parameters['C_s'],
        np.ones(n_l) * parameters['C_t']
    ])
    labels = np.concatenate([slabels, tlabels])
    
    # Initialize variables
    obj = []
    M = np.sqrt(parameters['lambda']) * np.ones((n, 1)) / np.sqrt(n)
   
    # MKL parameters
    lp_param = {
        'svm_C': 1,
        'd_norm': 1,
        'degree': parameters.get('mkl_degree', 1),
        'weight': weight
    }
    
    for i in range(MAX_ITER):
        # print(f"\tIter #{i+1:2d}:")
        # This needs to be implemented (see note below)
        theta, tmp_model, tmp_obj = LpMKL_HFA(labels, K_root, M, lp_param)
        obj.append(tmp_obj)
        model = tmp_model
        support_vector_indices = model.support_
        alpha = np.zeros(n)
        alpha[support_vector_indices] = np.abs(model.dual_coef_.flatten())

        # Check convergence
        if (i > 0) and ((abs(obj[i] - obj[i-1]) <= tau * abs(obj[i])) or (i == MAX_ITER - 1)):
            break
        
        # Update Hvs
        y_alpha = labels * alpha
        temp_beta = K_root @ y_alpha
        new_M = np.sqrt(parameters['lambda']) * temp_beta / np.sqrt(temp_beta.T @ temp_beta)
        M = np.column_stack([M, new_M]) if M.ndim > 1 else np.vstack([M, new_M]).T


    # Compute final H matrix
    tmp_M = M * np.sqrt(theta) if M.ndim > 1 else M * np.sqrt(theta[0])
    H = tmp_M @ tmp_M.T
    
    return model, H, obj
