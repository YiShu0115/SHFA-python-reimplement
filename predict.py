import numpy as np
def predict_f(kernel, model, Us, labels, d, rho, K_root, L_t_inv):
    n, n_t = L_t_inv.shape
    n_s = n - n_t
    
    # Get alpha values from model
    alpha = np.zeros(n)
    sv_indices = model.support_
    sv_coef = model.dual_coef_
    alpha[sv_indices] = np.abs(sv_coef[0])
    
    dec_values = np.zeros(kernel.shape[0])
    
    for i in range(labels.shape[1]):
        y_alpha = alpha * labels[:,i]
        y_alpha_t = y_alpha[n_s:]
        U_i=Us[:,i].reshape(-1,1)
        # Compute the decision components
        tmp = kernel @ L_t_inv.T @ U_i @ (U_i.T @ K_root)
        dec = tmp @ y_alpha + kernel @ y_alpha_t + np.sum(y_alpha)
        
        dec_values += d[i] * dec
    
    dec_values = dec_values / rho
    
    return dec_values