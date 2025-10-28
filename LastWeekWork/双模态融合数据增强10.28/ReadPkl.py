import pickle
import numpy as np


def inspect_structure(obj, indent_level=0):
    prefix = "  " * indent_level

    if isinstance(obj, dict):
        print(f"{prefix}[dict] 包含 {len(obj)} 个键:")
        for key, value in obj.items():
            print(f"{prefix}  - 键: '{key}'", end=" -> ")
            inspect_structure(value, indent_level + 1)

    elif isinstance(obj, list):
        print(f"{prefix}[list] 包含 {len(obj)} 个元素:")
        if len(obj) > 0:
            print(f"{prefix}  - 元素 0 的类型:", end=" ")
            inspect_structure(obj[0], indent_level + 1)
        if len(obj) > 6:
            print(f"{prefix}  (...等等，共 {len(obj)} 个)")

    elif isinstance(obj, np.ndarray):
        print(f"{prefix}[numpy.ndarray] 形状(Shape): {obj.shape}, 类型(Dtype): {obj.dtype}")

    elif isinstance(obj, (int, float, str, bool, type(None))):
        print(f"{prefix}[{type(obj).__name__}] 值: {obj}")

    else:
        print(f"{prefix}[{type(obj).__name__}] (一个对象)")


pkl_path = r'augmented_dataset_5fold.pkl'

try:
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # print("=" * 80)
    # print("完整数据结构:")
    # print("=" * 80)
    # inspect_structure(data)

    print("\n" + "=" * 80)
    print("查看 augmented_folds 中每个 fold 的 best_params:")
    print("=" * 80)

    # 检查是否存在 augmented_folds
    if isinstance(data, dict) and 'augmented_folds' in data:
        augmented_folds = data['augmented_folds']

        if isinstance(augmented_folds, list):
            print(f"\naugmented_folds 包含 {len(augmented_folds)} 个 fold\n")

            for i, fold in enumerate(augmented_folds):
                print(f"\n--- Fold {i + 1} ---")
                if isinstance(fold, dict) and 'best_params' in fold:
                    print(f"best_params: {fold['best_params']}")
                else:
                    print(f"该 fold 的结构: {type(fold)}")
                    if isinstance(fold, dict):
                        print(f"包含的键: {list(fold.keys())}")
        else:
            print(f"\naugmented_folds 不是列表，而是: {type(augmented_folds)}")
    else:
        print("\n数据中未找到 'augmented_folds' 键")
        if isinstance(data, dict):
            print(f"可用的键有: {list(data.keys())}")

except Exception as e:
    print(f"\n读取或探查时出错: {e}")
    import traceback

    traceback.print_exc()
