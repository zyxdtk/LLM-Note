from mlx_lm import load, generate

# model  list 
# mlx-community/Qwen3-0.6B-4bit-DWQ-05092025
# models--mlx-community--Qwen3-8B-4bit-DWQ-053125
# mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit-DWQ 16g内存跑不动
# mlx-community/Qwen3-14B-4bit-DWQ-053125

model, tokenizer = load("mlx-community/Qwen3-14B-4bit-DWQ-053125")

prompt = "帮我写一个数独游戏"
max_tokens = 10000

messages = [{"role": "system", "content": "你是一个无所不能的人工智能，你会帮用户完成各种任务"}, {"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True
)

text = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=True)