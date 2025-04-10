import numpy as np
from sklearn.svm import OneClassSVM
from MKL_shfa import LpMKL_SHFA 

def train_shfa_pnorm(slabels, tlabels, K, K_root, features, parameters):
    """
    Python implementation of SHFA classifier using scikit-learn
    
    Parameters:
    - slabels: source domain labels (n_s,)
    - tlabels: target domain labels (n_l,)
    - K: kernel matrix (n, n)
    - K_root: sqrt of kernel matrix (n, n)
    - features: augmented features (dim, n)
    - parameters: dict with keys:
        * C_s, C_t, C_x: regularization params
        * sigma: scaling param
        * mkl_degree: p-norm degree
        * hfa_iter: max iterations
        * hfa_tau: convergence threshold
        * upper_ratio, lower_ratio: label ratio bounds
    
    Returns:
    - model: trained SVC
    - Hvs: learned Hvs (n, m)
    - labels: all labels (n, m)
    - d: kernel coeffs
    - rho: margin param
    - obj: objective values
    """
    # Initialize parameters

    MAX_ITER = parameters.get('hfa_iter', 20)
    tau = parameters.get('hfa_tau', 1e-3)
    n_s = len(slabels)
    n_l = len(tlabels)
    n = K.shape[0]
    n_u = n - n_s - n_l
    upper_r = parameters['upper_ratio']
    lower_r = parameters['lower_ratio']
    
    # Compute weights
    weight = np.concatenate([
        np.ones(n_s) / parameters['C_s'],
        np.ones(n_l) / parameters['C_t'],
        np.ones(n_u) / parameters.get('C_x', 1)
    ])
    
    # ===== Initial SVM Training =====
    # tlabels = tlabels.reshape(-1, 1)
    # slabels = slabels.reshape(-1, 1)
    # assert tlabels.shape == (n_l, 1), "tlabels 必须是列向量"

    # 修正2：正确的核矩阵切片 (注意Python是0-based)
    Q_l = (K[n_s:n_s+n_l, n_s:n_s+n_l] + 1) * (tlabels @ tlabels.T) + np.diag(weight[n_s:n_s+n_l])
    assert np.allclose(Q_l, Q_l.T), "Q_l 必须对称"
    # 修正3：参数对齐MATLAB的 -s 2 -t 4 -n 1/n_l
    model = OneClassSVM(
        kernel='precomputed',
        nu=1/n_l,          # 对应 -n 参数
        shrinking=True,    # 默认启用，对应 -h
        tol=1e-3           # 默认值，对应 -e
    )

    # 修正4：准备LIBSVM格式的输入数据
    # train_data = np.column_stack((np.arange(1, n_l+1), Q_l.ravel()))  # 添加索引列 (1-based)
    model.fit(Q_l)  # 不需要标签

    # 修正5：获取alpha值 (处理边界支持向量)
    alpha = np.zeros(n_l)
    alpha[model.support_] = np.abs(model.dual_coef_.flatten())    
    # 边界支持向量的alpha应为 1/(nu*n_l)
    boundary_mask = model.dual_coef_.flatten() == 0
    alpha[model.support_[boundary_mask]] = 1/(model.nu * n_l)
    y_alpha = tlabels * alpha
    
    # Predict unlabeled
    decs = (K[n_s+n_l:, n_s:n_s+n_l] + 1) @ y_alpha
    y_u = (decs > 0).astype(int)
    sind = np.argsort(decs)[::-1]
    
    # Adjust label ratios
    if np.sum(y_u) > upper_r * n_u:
        y_u[sind[int(upper_r * n_u):]] = 0
    elif np.sum(y_u) < lower_r * n_u:
        y_u[sind[:int(lower_r * n_u)]] = 1
    
    y_u = 2 * y_u - 1
    labels = np.concatenate([slabels.ravel() , tlabels.ravel() , y_u.ravel()])
    labels = labels.reshape(-1, 1)
    # ===== SHFA Training =====
    obj = []
    Hvs = np.sqrt(parameters['sigma']) * np.ones((n, 1)) / np.sqrt(n)
    
    lp_param = {
        'd_norm': 1,
        'degree': parameters['mkl_degree'],
        'weight': weight
    }
    
    for i in range(MAX_ITER):
        # print(f'\tIter #{i+1:2d}:')
        
        d, tmp_model, tmp_obj, kernel = LpMKL_SHFA(labels, K, K_root, Hvs, lp_param)
        
        obj.append(tmp_obj)
        model = tmp_model
        
        if i > 0:
            print(f'obj = {obj[i]:.15f}, abs(obj({i+1}) - obj({i})) = {abs(obj[i] - obj[i-1]):.15f}')
        else:
            print(f'obj = {obj[i]:.15f}')
        
        alpha = np.zeros(n)
        alpha[model.support_] = np.abs(model.dual_coef_[0])
        
        if (i > 0 and abs(obj[i] - obj[i-1]) <= tau * abs(obj[i])) or (i == MAX_ITER - 1):
            break
        
        # ===== Update Hvs =====
        dim = features.shape[0]
        ax = features.T * alpha.reshape(-1, 1)
        axu = ax[n_s+n_l:]
        
        # Positive/negative sides
        psind = np.argsort(axu, axis=0)[::-1]
        nsind = np.argsort(-axu, axis=0)[::-1]
        sind = np.hstack([psind, nsind])
        y_u = np.hstack([axu > 0, -axu > 0])
        
        # Adjust label ratios
        for j in range(2*dim):
            y = y_u[:, j]
            if np.sum(y) > upper_r * n_u:
                y_u[sind[int(upper_r * n_u):, j], j] = 0
            elif np.sum(y) < lower_r * n_u:
                y_u[sind[:int(lower_r * n_u), j], j] = 1
        
        y_u = 2 * y_u - 1

        y = np.vstack([
            np.tile(np.concatenate([slabels, tlabels]).reshape(-1, 1), (1, 2*dim)),
            y_u
        ])
        v = np.hstack([ax, -ax]) * y
        values = np.abs(np.sum(v, axis=0))
        mind = np.argmax(values)
        y = y[:, mind]
        
        # Update Hvs
        y_alpha = y.flatten() * alpha
        temp_beta = K_root @ y_alpha
        scaled_beta = (np.sqrt(parameters['sigma']) * temp_beta / np.sqrt(temp_beta.T @ temp_beta))
        # 确保维度匹配
        if Hvs.ndim == 2 and scaled_beta.ndim == 1:
            scaled_beta = scaled_beta.reshape(-1, 1)  # 转为列向量
        elif Hvs.ndim == 1 and scaled_beta.ndim == 2:
            Hvs = Hvs.reshape(-1, 1)  # 或者将Hvs转为列向量

        # 水平堆叠
        Hvs = np.hstack([Hvs, scaled_beta])
        labels = np.hstack([labels, y.flatten().reshape(-1, 1)])
    
    # Calculate rho
    rho = kernel @ alpha
    rho = np.mean(rho[alpha > 0])
    
    return model, Hvs, labels, d, rho, obj

