import numpy as np
from sklearn import svm

def LpMKL_SHFA(labels, K, K_root, Hvs, param):
    """
    Python implementation of LpMKL_H_labels_v4 using scikit-learn
    
    Parameters:
    - labels: (n_samples, n_basekernels) label matrix
    - K: (n_samples, n_samples) kernel matrix
    - K_root: (n_samples, n_samples) sqrt of kernel matrix
    - Hvs: (n_samples, n_basekernels) matrix
    - param: dict with keys:
        * degree: p-norm degree (default=1)
        * d_norm: lp-norm constraint
        * weight: sample weights
        * d: initial coefficients (optional)
    
    Returns:
    - coefficients: learned kernel coefficients
    - model: trained SVC model
    - last_obj: final objective value
    - kernel: combined kernel matrix
    """
    n_samples, n_basekernels = Hvs.shape
    # labels = labels.reshape(-1,1)

    
    # Initialize parameters
    degree = param.get('degree', 1)
    d_norm = param.get('d_norm', 1)
    MAX_ITER = 20
    tau = 0.001
    
    # Initialize coefficients
    if 'd' in param:
        d = param['d']
        d = d_norm * d / (np.sum(d**degree)**(1/degree))
        coefficients = np.zeros(n_basekernels)
        coefficients[:len(d)] = d
    else:
        coefficients = d_norm * np.ones(n_basekernels) * (1/n_basekernels)**(1/degree)
    
    obj = []
    # First iteration
    model, obj_val, wn, kernel = return_alpha(K,K_root, Hvs, labels, coefficients, param)
    obj.append(obj_val)

    # Main optimization loop
    for i in range(1,MAX_ITER):
        # Update coefficients
        wnp = wn**(2/(degree+1))
        eta = (np.sum(wnp**degree))**(1/degree)
        coefficients = d_norm * wnp / eta
        model, obj_val, wn, kernel = return_alpha(K, K_root, Hvs, labels, coefficients, param)
        obj.append(obj_val)
        
        # if i == 0:
        #     print(f'Initial obj = {obj_val}')
        # else:
        #     print(f'Iter {i}: obj = {obj_val}, Δ = {abs(obj[i]-obj[i-1]):.15f}')
        
        if abs(obj[i]-obj[i-1]) <= tau*abs(obj[i]):
            break
        

    
    return coefficients, model, obj[-1], kernel

def return_alpha(K, K_root, Hvs, labels, coefficients, param):
    """
    Subfunction to compute alpha and related quantities
    """
    n = Hvs.shape[0]
    # Compute combined kernel
    # print('start calculating kernels...')
    kernel = sumkernels(K, K_root, Hvs, labels, coefficients) + np.diag(param['weight'])
    # print('end calculating kernels...')
    model = svm.OneClassSVM(kernel='precomputed', nu=1/n)

    # Prepare training data (note: y labels not needed for OneClassSVM in sklearn)
    train_data = kernel  # 1-based索引

    # Fit the model
    model.fit(train_data)

    # 获取支持向量信息
    idx = model.support_        # 支持向量索引 (0-based)
    alpha = np.abs(model.dual_coef_).flatten()  # 绝对值alpha值 (形状: n_sv)

        # 确保维度匹配
    if len(alpha) != len(idx):
        raise ValueError(f"alpha长度({len(alpha)})与支持向量数量({len(idx)})不匹配")

    # 计算 SU = K_root[idx, :] @ Hvs
    SU = K_root[idx, :] @ Hvs  # 形状: (n_sv, r)

    # 计算 AY = alpha * labels[idx] (使用广播)
    AY = alpha[:, np.newaxis] * labels[idx]  # 形状: (n_sv, r)

    # 计算 q 的三部分
    part1 = np.sum(SU * AY, axis=0)**2  # 形状: (r,)
    part2 = np.sum((K_root[:, idx] @ AY)**2, axis=0)  # 形状: (r,)
    part3 = np.sum(AY, axis=0)**2  # 形状: (r,)

    q = part1 + part2 + part3  # 形状: (r,)
    # 确保所有数组维度正确
    q_flat = q.flatten()  # 确保形状为 (r,)
    weight_sv = param['weight'][idx]   # 确保形状为 (r,)

    obj =  -0.5 * (np.sum(coefficients * q_flat) + alpha.T @ (alpha * weight_sv))
    wn = coefficients * np.sqrt(q)  # 形状 (r,)
    return model, obj, wn, kernel

def sumkernels(K, K_root, Hvs, labels, d):

    sqrt_d = np.sqrt(d).reshape(-1, 1)
    wlabels = labels.T * sqrt_d

    item1 = (K_root.T @ Hvs * wlabels.T) @ (K_root.T @ Hvs * wlabels.T).T
    item2 = (K + 1) * (wlabels.T @ wlabels)

    # 单行计算，减少临时数组
    return item1+item2
