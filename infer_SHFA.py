from data.data_preprocess import load_arrays_from_json
import os
import numpy as np

import numpy as np
from scipy.linalg import sqrtm, pinv
from GetKernel import get_kernel

from predict import predict_f
from train_shfa import train_shfa_pnorm

# Load data
loaded_data = load_arrays_from_json('./data/arrays.json')
source_features = loaded_data['source_features']
target_features = loaded_data['target_features']
test_features = loaded_data['test_features']
source_labels = loaded_data['source_labels']
target_labels = loaded_data['target_labels']
test_labels = loaded_data['test_labels']
target_features = np.hstack([target_features, test_features])

categories = os.listdir("./data/office31/amazon" )
unique_labels = sorted(list(set(categories))) 


#Compute kernel
param = {
    'C_s': 5,
    'C_t': 1,
    'C_x': 1e-3,
    'sigma': 100,
    'mkl_degree': 2,
    'ratio_var': 0,
    'hfa_iter': 20,
    'hfa_tau': 0.001
}
source_features = np.squeeze(source_features)
target_features = np.squeeze(target_features)
test_features = np.squeeze(test_features)

kparam = {'kernel_type': 'gaussian'}
K_s, param_s = get_kernel(source_features,kparam)
kparam = {'kernel_type': 'gaussian'}
K_t, param_t = get_kernel(target_features,kparam)

K_s_root = np.real(sqrtm(K_s))
K_t_root = np.real(sqrtm(K_t))
n_s = K_s.shape[0]
n_t = K_t.shape[0]

K = np.block([[K_s, np.zeros((n_s, n_t))], [np.zeros((n_t, n_s)), K_t]])
K_root = np.block([[K_s_root, np.zeros((n_s, n_t))], [np.zeros((n_t, n_s)), K_t_root]])

K_t_root_inv = np.real(pinv(K_t_root))
L_t_inv = np.vstack([np.zeros((n_s, n_t)), np.eye(n_t)]) @ K_t_root_inv

# Kernel decomposition for inference
aug_features = np.real(sqrtm((1 + param['sigma']) * K + np.ones(K.shape)))
K_test = get_kernel(test_features, target_features, param_t)[0]

n_test_samples = test_features.shape[1]
n_categories = len(categories)-1
dec_values = np.zeros((n_test_samples, n_categories))

# Train one-versus-all classifiers
for c in range(n_categories):
    cat_name = unique_labels[c+1]
    print(f'-- Class {c+1}: {cat_name}')
    
    # Prepare binary labels
    source_labels_2 = 2 * (source_labels == c+1) - 1
    target_labels_2 = 2 * (target_labels == c+1) - 1
    
    # Set ratio of positive samples
    ratio = np.sum(test_labels == c+1) / len(test_labels)
    param['upper_ratio'] = ratio
    param['lower_ratio'] = ratio
    print(K.shape)
    
    # Training
    model, Us, labels, coefficients, rho, obj = train_shfa_pnorm(
        source_labels_2, target_labels_2, K, K_root, aug_features, param)
    
    #Test
    dec_values[:, c] = predict_f(
        K_test, model, Us, labels, coefficients, rho, K_root, L_t_inv)
    
dec_values_normalized = np.zeros_like(dec_values)
for c in range(n_categories):
    mean = np.mean(dec_values[:, c])
    std = np.std(dec_values[:, c])
    dec_values_normalized[:, c] = (dec_values[:, c] - mean) / (std + 1e-8)  # 避免除零

# =============================================
# Display results
predict_labels = np.argmax(dec_values_normalized, axis=1) + 1  # +1 for 1-based indexing
acc = np.sum(predict_labels == test_labels) / len(test_labels)
print(f'SHFA accuracy = {acc:.6f}')
