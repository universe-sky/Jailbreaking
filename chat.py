from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

def chat_with_llama():
    # 手动设置使用的 GPU
    gpu_id = input("请输入要使用的 GPU 编号（例如 0 或 1）：")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    model_path = "/data/home/jutj/Llama-3.1-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).cuda()

    print("Llama3 chatbot is ready! Type 'exit' to quit.")
    print("=======================")

    hidden_states_file = os.path.expanduser("hidden_states.txt")

    # 创建文件存储中间层数据，确保文件为空
    with open(hidden_states_file, "w") as f:
        f.write("")

    # 读取问题列表
    with open("question.txt", "r") as f:
        questions = [line.strip() for line in f.readlines()]

    for user_input in questions:
        # 更新会话历史
        conversation_history = f"User: {user_input}\nAssistant: "

        # 编码输入文本
        inputs = tokenizer(conversation_history, return_tensors="pt").to('cuda')

        # 用于捕获中间层数据的列表
        captured_hidden_states = []

        def save_hidden_states(module, input, output):
            """捕获每一层的中间状态."""
            if isinstance(output, tuple):
                output = output[0]  # 如果是 tuple，取第一个元素
            captured_hidden_states.append(output)  # 保存到列表

        # 注册 hook 捕获前 32 层的中间层数据
        hook_handles = []
        for i, layer in enumerate(model.model.layers):
            if i < 32:  # 只捕获前32层
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
                f.write(f"Question: {user_input}\n")  # 记录问题

                # 遍历 32 层，保存每层最后一个 token 的向量数据
                for layer_idx in range(len(captured_hidden_states)):  # 确保只遍历捕获到的层
                    if layer_idx < 32:  # 确保不超过 32 层
                        last_token_vector = captured_hidden_states[layer_idx][:, -1, :].cpu().detach().numpy().tolist()
                        f.write(f"Last token vector from layer {layer_idx}: {last_token_vector}\n")

                f.write("=" * 50 + "\n")  # 添加分隔符

        # 解码模型输出，获取回复
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取 Assistant 部分的回复
        if "Assistant: " in response:
            response = response.split("Assistant: ")[-1].strip()
        else:
            response = response.strip()

        # 打印回复
        print(f"Llama3: {response}")

if __name__ == "__main__":
    chat_with_llama()
