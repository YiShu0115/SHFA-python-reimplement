from data.data_preprocess import load_arrays_from_json
import os
import numpy as np

import numpy as np
from scipy.linalg import sqrtm
from GetKernel import get_kernel

from train_hfa import train_hfa_mkl

# Load data
loaded_data = load_arrays_from_json('./data/arrays.json')
source_features = loaded_data['source_features']
target_features = loaded_data['target_features']
test_features = loaded_data['test_features']
source_labels = loaded_data['source_labels']
target_labels = loaded_data['target_labels']
test_labels = loaded_data['test_labels']

categories = os.listdir("./data/office31/amazon" )
unique_labels = sorted(list(set(categories))) 

#Compute kernel
kparam = {'kernel_type': 'gaussian'}
K_s, param_s = get_kernel(source_features,kparam)
kparam = {'kernel_type': 'gaussian'}
K_t, param_t = get_kernel(target_features,kparam)

K_s_root = np.real(sqrtm(K_s))
K_t_root = np.real(sqrtm(K_t))
n_s = K_s.shape[0]
n_t = K_t.shape[0]
n = n_s + n_t

K_root = np.block([[K_s_root, np.zeros((n_s, n_t))], [np.zeros((n_t, n_s)), K_t_root]])
K_t_root_inv = np.real(np.linalg.pinv(K_t_root))
L_t_inv = np.vstack([np.zeros((n_s, n_t)), np.eye(n_t)]) @ K_t_root_inv


K_test,param_k = get_kernel(test_features, target_features, param_t)

# Train one-versus-all classifiers
param = {
    'C_s': 1,
    'C_t': 1,
    'lambda': 100,
    'mkl_degree': 1
}

n_classes = len(unique_labels)-1
dec_values = np.zeros((test_labels.shape[0], n_classes))
for c in range(n_classes):
    class_name = unique_labels[(c+1)]
    print(f'-- Class {(c+1)}: {class_name}')
    source_binary_labels = np.where(source_labels == (c+1), 1, -1)
    target_binary_labels = np.where(target_labels == (c+1), 1, -1)

    # Training 
    model, H, obj = train_hfa_mkl(source_binary_labels, target_binary_labels, K_root, param)
    
    # Testing
    rho = model.intercept_[0] * model.classes_[1]
    y_alpha = np.zeros(n)
    y_alpha[model.support_] = model.dual_coef_.ravel() * model.classes_[1]
    y_alpha_t = y_alpha[n_s:]
    tmp = K_test @ L_t_inv.T @ H @ K_root
    dec_values[:, c] = tmp @ y_alpha + K_test @ y_alpha_t - rho

dec_values_normalized = np.zeros_like(dec_values)
for c in range(n_classes):
    mean = np.mean(dec_values[:, c])
    std = np.std(dec_values[:, c])
    dec_values_normalized[:, c] = (dec_values[:, c] - mean) / (std + 1e-8)

# =========================================================================
# Display results
predict_labels =np.argmax(dec_values_normalized, axis=1) + 1 
test_labels = np.squeeze(test_labels)
acc = np.sum(predict_labels == test_labels) / len(test_labels) 
print(f'HFA accuracy = {acc:.6f}')