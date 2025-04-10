import os
import numpy as np
import cv2
from skimage.io import imread
from skimage.color import rgb2gray
from sklearn.cluster import MiniBatchKMeans
import json
import base64

def load_images_with_vq(dataset_path, random_seed=1, samples_per_class=20, remain=False, n_clusters=800):
    """
    使用SIFT特征 + 向量量化生成图像特征
    参数：
        n_clusters: 码本大小（聚类中心数）
    """
    np.random.seed(random_seed)
    sift = cv2.SIFT_create()  # 修改为SIFT特征提取器
    all_descriptors = []
    data_dict = {'train': [], 'remain': []}

    # 第一阶段：收集所有SIFT描述子
    print("收集SIFT描述子...")
    for label in sorted(os.listdir(dataset_path)):
        label_path = os.path.join(dataset_path, label)
        if not os.path.isdir(label_path):
            continue
            
        all_images = [f for f in os.listdir(label_path) if f.endswith(('.jpg', '.png'))]
        selected_images = np.random.choice(all_images, samples_per_class, replace=False)
        unselected_images = [img for img in all_images if img not in selected_images] if remain else []
        
        for img_file in all_images:
            img_path = os.path.join(label_path, img_file)
            try:
                image = imread(img_path)
                gray = rgb2gray(image) * 255
                gray = gray.astype('uint8')
                
                # 提取SIFT描述子（128维）
                kp, desc = sift.detectAndCompute(gray, None)
                if desc is not None and len(desc) > 10:
                    all_descriptors.extend(desc)
                    key = 'train' if img_file in selected_images else 'remain'
                    data_dict[key].append((desc, label))
            except Exception as e:
                print(f"跳过图像 {img_path}: {e}")

    # 训练码本
    print("训练码本...")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_seed)
    kmeans.fit(all_descriptors)
    
    def generate_features(data_list):
        features, labels = [], []
        for desc, label in data_list:
            if desc is None or len(desc) == 0:
                hist = np.zeros(n_clusters, dtype=np.float64)
            else:
                # 关键修复步骤（四重保险）
                desc = np.ascontiguousarray(desc, dtype=np.float64)  # 1. 强制连续内存+类型
                desc = desc.reshape(-1, 128)                      # 2. 确保二维结构
            
                # 安全预测（封装预测过程）
                try:
                    visual_words = predict_with_type_check(kmeans, desc)
                    hist = np.bincount(visual_words, minlength=n_clusters).astype(np.float64)
                    # hist /= (hist.sum() + 1e-6)
                except Exception as e:
                    print(f"预测失败：{str(e)}")
                    hist = np.zeros(n_clusters, dtype=np.float64)
            
            features.append(hist)
            labels.append(label)
        
        return np.array(features, dtype=np.float32), np.array(labels)

    def predict_with_type_check(model, X):
        """类型安全的预测封装"""
        X = np.asarray(X, dtype=np.float64, order='C')
        if not X.flags['C_CONTIGUOUS']:
            X = np.ascontiguousarray(X)
        return model.predict(X)
    
    train_features, train_labels = generate_features(data_dict['train'])
    if remain:
        remain_features, remain_labels = generate_features(data_dict['remain'])
        return train_features, train_labels, remain_features, remain_labels
    return train_features, train_labels

def save_arrays_to_json(file_path, **arrays):
    """
    保存多个NumPy数组到单个JSON文件
    参数：
        file_path: 保存路径（如 'data.json'）
        **arrays: 键值对形式的数组（如 arr1=array1, arr2=array2）
    """
    data = {}
    for name, array in arrays.items():
        # 将NumPy数组转为二进制+Base64字符串
        data[name] = {
            'dtype': str(array.dtype),
            'shape': array.shape,
            'data': base64.b64encode(array.tobytes()).decode('utf-8')
        }
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
        
def load_arrays_from_json(file_path):
    """
    从JSON文件加载多个NumPy数组
    返回：包含数组的字典（键为保存时的名称）
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    arrays = {}
    for name, array_data in data.items():
        # 从Base64字符串重建数组
        buffer = base64.b64decode(array_data['data'])
        arrays[name] = np.frombuffer(
            buffer, 
            dtype=np.dtype(array_data['dtype'])
        ).reshape(array_data['shape'])
    
    return arrays

if __name__ == '__main__':
    dataset_path1 = "./office31/amazon" 
    amazon_feature,amazon_label = load_images_with_vq(dataset_path1)

    dataset_path2 = "./office31/dslr"  
    dslr_feature, dslr_label, dslr_remain_feature, dslr_remain_label = load_images_with_vq(dataset_path2,samples_per_class=3,remain=True, n_clusters=600)

    dataset_path3 = "./office31/webcam"  
    webcam_feature,webcam_label = load_images_with_vq(dataset_path3,samples_per_class=8)


    save_arrays_to_json('arrays.json', 
                        amazon_feature = amazon_feature, amazon_label = amazon_label,
                        dslr_feature=dslr_feature, dslr_label=dslr_label,
                        dslr_remain_feature=dslr_remain_feature, dslr_remain_label= dslr_remain_feature,
                        webcam_feature=webcam_feature, webcam_label=webcam_label)