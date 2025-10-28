import pickle
import numpy as np

# 这是您新生成的文件
pkl_path = r'augmented_dataset_5fold.pkl'

print(f"--- 正在读取文件: {pkl_path} ---")

"""读取pkl文件"""
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

"""
新文件 'augmented_dataset_5fold.pkl' 的顶层结构包含以下键 (Keys):
- 'augmented_folds': 一个包含5折数据的列表 (list)
- 'fold_results':    一个包含5折调优结果的列表 (list)
- 'x_test':          包含测试集数据的字典 (dict)
- 'num_classes':     类别数量 (int)
- 'n_splits':        折数 (int)
"""
print(f"文件顶层键 (Keys): {data.keys()}")


# --- 我们以第1折 (索引为0) 为例来查看数据 ---
print("\n--- 正在检查 [第 1 折] (索引 0) 的数据 ---")

# 从 'augmented_folds' 列表中取出第1折的数据
fold_1_data = data['augmented_folds'][0]
print(f"第1折的键 (Keys): {fold_1_data.keys()}")


"""取出第1折的 [增强后训练集]"""
# 注意键已变为 'train_spec', 'train_env', 'train_labels'
x_train_spec_f1 = fold_1_data['train_spec']  # 频谱特征
x_train_env_f1 = fold_1_data['train_env']    # 包络特征
y_train_f1 = fold_1_data['train_labels']      # 训练集标签

"""取出第1折的 [验证集]"""
x_val_spec_f1 = fold_1_data['val_spec']
x_val_env_f1 = fold_1_data['val_env']
y_val_f1 = fold_1_data['val_labels']


"""取出 [测试集] (测试集结构保持不变)"""
x_test_spec = data['x_test']['spec']  # 频谱特征
x_test_env = data['x_test']['env']  # 包络特征


"""打印形状"""
print("\n--- [第 1 折] 增强后训练数据 (Augmented Train) ---")
print(f"x_train 'spec' Shape: {x_train_spec_f1.shape}, Dtype: {x_train_spec_f1.dtype}")
print(f"x_train 'env'  Shape: {x_train_env_f1.shape}, Dtype: {x_train_env_f1.dtype}")
print(f"y_train        Shape: {y_train_f1.shape}, Dtype: {y_train_f1.dtype}")

print("\n--- [第 1 折] 验证数据 (Validation) ---")
print(f"x_val 'spec' Shape: {x_val_spec_f1.shape}, Dtype: {x_val_spec_f1.dtype}")
print(f"x_val 'env'  Shape: {x_val_env_f1.shape}, Dtype: {x_val_env_f1.dtype}")
print(f"y_val        Shape: {y_val_f1.shape}, Dtype: {y_val_f1.dtype}")

print("\n--- [全局] 测试数据 (Test) ---")
print(f"x_test 'spec' Shape: {x_test_spec.shape}, Dtype: {x_test_spec.dtype}")
print(f"x_test 'env'  Shape: {x_test_env.shape}, Dtype: {x_test_env.dtype}")

# 检查一下类别平衡的效果
print("\n--- [第 1 折] 增强后训练集 类别平衡检查 ---")
unique_classes, counts = np.unique(y_train_f1, return_counts=True)
for cls, count in zip(unique_classes, counts):
    print(f"  类别 {cls}: {count} 个样本")