import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def train_layerwise_classifiers(hidden_states_file, labels_file):
    # 加载 hidden states 和标签
    hidden_states, labels = load_hidden_states_and_labels(hidden_states_file, labels_file)

    # 将标签转换为 0 和 1
    label_mapping = {"legal": 0, "illegal": 1}
    label_indices = [label_mapping[label] for label in labels]

    print(f"Label distribution: {np.bincount(label_indices)}")  # 打印标签分布

    layer_info_gains = []

    for layer_idx in range(32):
        print(f"Processing layer {layer_idx}...")

        # 提取当前层的隐状态 (只取最后一个token)
        layer_hidden_states = hidden_states[:, layer_idx, -1, :]  # shape: (num_samples, hidden_size)

        # 划分训练集和测试集，确保每个类别都有样本
        X_train, X_test, y_train, y_test = train_test_split(
            layer_hidden_states, label_indices, test_size=0.2, 
            random_state=42, stratify=label_indices
        )

        print(f"Layer {layer_idx} hidden states shape: {layer_hidden_states.shape}")
        print(f"Training labels distribution for layer {layer_idx}: {np.bincount(y_train)}")
        print(f"Testing labels distribution for layer {layer_idx}: {np.bincount(y_test)}")

        # 1. 模拟 H(y|b) 使用全零输入
        model_null = LogisticRegression(max_iter=1000)
        X_train_null = np.zeros_like(X_train)  # 全零输入
        model_null.fit(X_train_null, y_train)

        y_pred_null = model_null.predict_proba(np.zeros_like(X_test))  # 全零测试输入
        H_yb = -np.mean([np.log2(y_pred_null[i][y_test[i]] + 1e-10) for i in range(len(y_pred_null))])

        # 2. 计算 H(y|x)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)

        H_yx = -np.mean([np.log2(y_pred[i][y_test[i]] + 1e-10) for i in range(len(y_pred))])

        # 3. 计算信息增益 Vi
        Vi = H_yb - H_yx
        layer_info_gains.append(Vi)

        print(f'Layer {layer_idx}: Hyb = {H_yb}, Hyx = {H_yx}, Vi = {Vi}')

    # 保存每层的信息增益图像到文件
    plt.plot(range(32), layer_info_gains, marker='o')
    plt.title("Layer-wise Information Gain (Vi)")
    plt.xlabel("Layer Index")
    plt.ylabel("Information Gain (Vi)")
    plt.grid(True)
    
    # 将图像保存为 PNG 文件
    plt.savefig("layer_info_gain.png")
    print("Information gain plot saved as layer_info_gain.png.")

def load_hidden_states_and_labels(hidden_states_file, labels_file):
    # 加载隐藏状态
    hidden_states = []
    with open(hidden_states_file, "r") as f:
        current_layer = []
        for line in f:
            line = line.strip()
            if line.startswith("Last token vector from layer"):
                hidden_state = np.array(eval(line.split(":", 1)[1].strip()))
                current_layer.append(hidden_state)
            elif line == "=" * 50:  # 表示问题结束
                hidden_states.append(np.array(current_layer))
                current_layer = []

    hidden_states = np.array(hidden_states)
    print(f"Loaded hidden states shape: {hidden_states.shape}")  # 检查形状

    # 加载标签
    with open(labels_file, "r") as f:
        labels = f.read().splitlines()

    return hidden_states, labels


if __name__ == "__main__":
    hidden_states_file = "hidden_states.txt"
    labels_file = "labels.txt"
    train_layerwise_classifiers(hidden_states_file, labels_file)
