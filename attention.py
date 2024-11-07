import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

def extract_attention_weights():
    # 手动设置使用的 GPU
    gpu_id = input("请输入要使用的 GPU 编号（例如 0 或 1）：")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    model_path = "/data/home/jutj/Llama-3.1-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).cuda()

    print("准备提取注意力权重！")

    # 读取问题列表
    with open("attention_question.txt", "r") as f:
        questions = [line.strip() for line in f.readlines()]

    # 创建文件存储注意力权重
    with open("attention.txt", "w") as f:
        for user_input in questions:
            # 编码输入文本
            inputs = tokenizer(user_input, return_tensors="pt").to('cuda')

            # 生成模型回复并捕获注意力权重
            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)  # 获取注意力权重
                attention_weights = outputs.attentions  # 获取所有层的注意力权重

            # 处理和保存注意力权重
            for layer_idx, layer_attention in enumerate(attention_weights):
                layer_attention = layer_attention.cpu().detach().numpy()  # 转换为 NumPy 数组
                
                # 记录每层的注意力权重
                for head_idx in range(layer_attention.shape[1]):  # 遍历每个注意力头
                    f.write(f"Layer {layer_idx + 1}, Head {head_idx + 1}:\n")
                    for i in range(layer_attention.shape[-1]):  # 遍历每个 token
                        attention_row = layer_attention[0, head_idx, i].tolist()  # 获取当前 token 的注意力权重
                        tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
                        token_str = tokens[i]
                        f.write(f"Token '{token_str}': Attention Weights: {attention_row}\n")
                    f.write("=================================\n")

            print(f"注意力权重已保存，问题: {user_input}")

if __name__ == "__main__":
    extract_attention_weights()
