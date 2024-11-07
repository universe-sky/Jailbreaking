import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def load_attention_data(filename):
    attention_data = {}
    with open(filename, 'r') as f:
        layer = None
        head = None
        tokens = []
        weights = []

        for line in f:
            line = line.strip()
            if line.startswith("Layer"):
                parts = line.split(',')
                layer = int(parts[0].split()[1])  # 'Layer 1' -> 1
                head = int(parts[1].split()[1].replace(':', '').strip())  # ' Head 1:' -> 1
                
                # 只处理 Layer 1, Head 1
                if layer == 1 and head == 1:
                    tokens = []
                    weights = []
                else:
                    continue  # 跳过其他层和头

            elif line.startswith("Token") and layer == 1 and head == 1:
                token_info = line.split(':')
                if len(token_info) < 3:  # 需要至少3个部分
                    print(f"Unexpected token line format: {line}")
                    continue
                
                token = token_info[0].split('\'')[1]  # 获取 token
                
                # 直接使用 token_info[2] 获取权重部分
                weight_values_str = token_info[2].strip().strip('[] ')  # 去掉 '[]' 和空格
                
                if weight_values_str:  # 确保不是空字符串
                    weight_values = list(map(float, weight_values_str.split(',')))  # 转换为浮点数
                    tokens.append(token)
                    weights.append(weight_values)
                    
                    # 打印获取到的token和权重
                    print(f"Token: {token}, Weights: {weight_values}")

            elif "=================================" in line:  # 忽略分隔行
                continue

        # 直接保存 Layer 1, Head 1 的数据
        if tokens and weights:  # 确保 tokens 和 weights 不是空
            attention_data[(1, 1)] = (tokens, np.array(weights))

    print(f"Loaded attention data: {attention_data}")  # 调试输出
    return attention_data

def plot_attention_heatmaps(attention_data):
    print('1')
    for (layer, head), (tokens, weights) in attention_data.items():
        print('2')
        if weights.size == 0:  # 如果权重矩阵为空，跳过
            print(f"No weights for Layer {layer}, Head {head}.")
            return  # 只读取一层，不继续绘图
        
        print(f"准备绘制图形: Layer {layer}, Head {head}")
        print(f"Tokens: {tokens}")
        print(f"Weights Shape: {weights.shape}")  # 输出权重的形状
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(weights, cmap='viridis', annot=True, fmt=".2f",
                    xticklabels=tokens, yticklabels=tokens)
        plt.title(f'Layer {layer}, Head {head}')
        plt.xlabel('Tokens')
        plt.ylabel('Tokens')
        
        # 保存图像而不是显示
        plt.savefig("attention.png")
        plt.close()  # 关闭当前图形以释放内存
        print("图片已保存")

# 示例用法
if __name__ == "__main__":
    attention_data = load_attention_data("attention.txt")
    if not attention_data:  # 检查是否有数据加载
        print("未加载到任何注意力数据。")
    else:
        plot_attention_heatmaps(attention_data)
