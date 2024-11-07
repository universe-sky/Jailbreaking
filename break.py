from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import numpy as np

# DeepFool 类的简单实现
class DeepFool:
    def __init__(self, model, x, num_classes=10, overshoot=0.02, max_iter=50):
        self.model = model
        self.x = x.float()  # 确保 x 是 FloatTensor
        self.num_classes = num_classes
        self.overshoot = overshoot
        self.max_iter = max_iter

    def perturb(self):
        x = self.x.clone().detach().requires_grad_(True)

        # 计算模型输出
        f_x = self.model(x)

        # 获取当前分类
        y_pred = f_x.argmax().item()
        w = np.zeros_like(x)
        r_tot = np.zeros_like(x)

        # 迭代优化
        for i in range(self.max_iter):
            f_x = self.model(x)
            classes = f_x.data.numpy().flatten()
            classes[y_pred] = -np.inf  # 忽略当前预测的类别
            k_idx = np.argmax(classes)  # 找到下一个类别

            # 计算梯度
            self.model.zero_grad()
            f_x[0, k_idx].backward()
            w_k = x.grad.data.cpu().numpy().copy()

            # 计算扰动
            r_i = (classes[k_idx] - classes[y_pred]) / np.linalg.norm(w_k)
            r_i = np.maximum(0, r_i) * w_k

            # 更新总扰动
            r_tot += r_i
            x = x + r_tot * (1 + self.overshoot)

            # 更新预测
            f_x = self.model(x)
            y_pred = f_x.argmax().item()
            if y_pred != k_idx:  # 如果成功越过边界，停止迭代
                break

        return torch.tensor(r_tot).to(x.device).float()  # 确保返回的是浮点数

def chat_with_llama_and_deepfool():
    # 手动设置使用的 GPU
    gpu_id = input("请输入要使用的 GPU 编号（例如 0 或 1）：")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    model_path = "/data/home/jutj/Llama-3.1-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).cuda()

    print("Llama3 chatbot is ready! Type 'exit' to quit.")
    print("=======================")

    hidden_states_file = os.path.expanduser("break_states.txt")
    new_hidden_states_file = os.path.expanduser("break_states_new.txt")

    # 创建文件存储中间层数据，确保文件为空
    with open(hidden_states_file, "w") as f:
        f.write("")
    with open(new_hidden_states_file, "w") as f:
        f.write("")

    # 读取越狱问题列表
    with open("break_question.txt", "r") as f:
        break_questions = [line.strip() for line in f.readlines()]

    for user_input in break_questions:
        # 更新会话历史
        conversation_history = f"User: {user_input}\nAssistant: "

        # 编码输入文本
        inputs = tokenizer(conversation_history, return_tensors="pt").to('cuda')

        # 用于捕获中间层数据的列表
        captured_hidden_states = []

        def save_hidden_states(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            captured_hidden_states.append(output)

        # 注册 hook 捕获前 32 层的中间层数据
        hook_handles = []
        for i, layer in enumerate(model.model.layers):
            if i < 32:
                hook_handles.append(layer.register_forward_hook(save_hidden_states))

        # 生成模型回复并捕获中间层数据
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=50,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=inputs.attention_mask,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

        # 移除 hook
        for handle in hook_handles:
            handle.remove()

        # 检查是否捕获到中间层数据
        if captured_hidden_states:
            with open(hidden_states_file, "a") as f:
                f.write(f"Question: {user_input}\n")
                for layer_idx in range(len(captured_hidden_states)):
                    if layer_idx < 32:
                        last_token_vector = captured_hidden_states[layer_idx][:, -1, :].cpu().detach().numpy().tolist()
                        f.write(f"Last token vector from layer {layer_idx}: {last_token_vector}\n")
                f.write("=" * 50 + "\n")

        # 解码模型输出
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Assistant: " in response:
            response = response.split("Assistant: ")[-1].strip()
        else:
            response = response.strip()

        # 打印回复
        print(f"Llama3: {response}")

        # 在应用 DeepFool 之前
        # 获取最后一层的向量并确保它是浮点数
        # 获取最后一层的向量并确保它是浮点数
        last_layer_vector = captured_hidden_states[31][:, -1, :].detach().cpu().numpy()  # 获取第32层最后一个token的向量

        # 使用 DeepFool 计算扰动
        last_layer_vector_tensor = torch.tensor(last_layer_vector).float().to('cuda')  # 保持浮点数

        perturbation = DeepFool(model, last_layer_vector_tensor).perturb()  # 确保是浮点数

        # 生成新的输入向量
        perturbed_vector = last_layer_vector + perturbation.detach().cpu().numpy()

        # 确保新的输入向量为整型（LongTensor），这里假设最后的perturbed_vector是token ID的形式
        perturbed_vector_tensor = torch.tensor(perturbed_vector).long().to('cuda')  # 转换为 LongTensor
        perturbed_vector_tensor = perturbed_vector_tensor.unsqueeze(0)  # 添加 batch 维度


        # 再次生成模型回复
        new_outputs = model.generate(
            perturbed_vector_tensor,
            max_new_tokens=50,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

        # 捕获新的中间层数据
        new_captured_hidden_states = []
        for i, layer in enumerate(model.model.layers):
            if i < 32:
                new_captured_hidden_states.append(layer.register_forward_hook(save_hidden_states))

        # 移除 hook
        for handle in new_captured_hidden_states:
            handle.remove()

        # 记录新的中间层数据
        if new_captured_hidden_states:
            with open(new_hidden_states_file, "a") as f:
                f.write(f"Perturbed Question: {user_input}\n")
                for layer_idx in range(len(new_captured_hidden_states)):
                    if layer_idx < 32:
                        new_last_token_vector = new_captured_hidden_states[layer_idx][:, -1, :].cpu().detach().numpy().tolist()
                        f.write(f"Last token vector from layer {layer_idx}: {new_last_token_vector}\n")
                f.write("=" * 50 + "\n")

        # 解码新的模型输出
        new_response = tokenizer.decode(new_outputs[0], skip_special_tokens=True)
        if "Assistant: " in new_response:
            new_response = new_response.split("Assistant: ")[-1].strip()
        else:
            new_response = new_response.strip()

        # 打印新的回复
        print(f"Perturbed Llama3: {new_response}")

if __name__ == "__main__":
    chat_with_llama_and_deepfool()
