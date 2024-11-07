import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

def visualize_hidden_states_with_labels():
    # 文件路径
    hidden_states_file = "hidden_states.txt"
    labels_file = "labels.txt"

    print("Checking for files...")
    
    # 检查文件是否存在
    if not os.path.exists(hidden_states_file):
        print(f"File {hidden_states_file} not found!")
        return

    if not os.path.exists(labels_file):
        print(f"File {labels_file} not found!")
        return

    print("Loading hidden states...")
    
    # 加载 hidden states
    hidden_states = []
    try:
        with open(hidden_states_file, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("Last token vector from layer 32:") and line.endswith("]"):
                    hidden_state = np.array(eval(line.split(":", 1)[1].strip()))
                    hidden_states.append(hidden_state)
    except Exception as e:
        print(f"Error while reading hidden states: {e}")
        return

    hidden_states = np.array(hidden_states)

    print(f"Extracted hidden states: {hidden_states.shape}")
    
    # 检查 hidden states 的形状
    if hidden_states.ndim == 3:
        hidden_states = hidden_states.reshape(hidden_states.shape[0], -1)

    print("Loading labels...")
    
    # 加载分类标签
    try:
        with open(labels_file, "r") as f:
            labels = f.read().splitlines()
    except Exception as e:
        print(f"Error while reading labels: {e}")
        return

    print(f"Total labels: {len(labels)}")

    # 检查 hidden states 和 labels 数量是否匹配
    if len(hidden_states) != len(labels):
        print(f"Mismatch between hidden states and labels: {len(hidden_states)} hidden states, {len(labels)} labels.")
        return

    # 将分类标签转换为 0 和 1
    label_mapping = {"legal": 0, "illegal": 1}
    try:
        label_indices = [label_mapping[label] for label in labels]
    except KeyError as e:
        print(f"Invalid label found: {e}")
        return

    print("Applying PCA...")
    
    # 使用 PCA 将 hidden states 降维到 2 维
    try:
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(hidden_states)
    except Exception as e:
        print(f"Error while applying PCA: {e}")
        return

    print("Applying t-SNE...")
    
    # 使用 t-SNE 将 hidden states 降维到 2 维
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=5)
        tsne_data = tsne.fit_transform(hidden_states)
    except Exception as e:
        print(f"Error while applying t-SNE: {e}")
        return

    print("Creating plots...")
    
    # 创建绘图
    try:
        plt.figure(figsize=(12, 5))

        # PCA 图
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], c=label_indices, cmap="coolwarm", marker="o", edgecolor="k", s=100)
        plt.title("PCA of Hidden States")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.colorbar(scatter, label="Class (0: legal, 1: illegal)")

        # t-SNE 图
        plt.subplot(1, 2, 2)
        scatter = plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=label_indices, cmap="coolwarm", marker="o", edgecolor="k", s=100)
        plt.title("t-SNE of Hidden States")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.colorbar(scatter, label="Class (0: legal, 1: illegal)")

        plt.tight_layout()
        plt.savefig("hidden_states_visualization.png")  # 保存图像
        print("Plots created successfully. Saved as hidden_states_visualization.png.")

    except Exception as e:
        print(f"Error while plotting: {e}")

if __name__ == "__main__":
    visualize_hidden_states_with_labels()