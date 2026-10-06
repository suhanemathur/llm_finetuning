import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="cuda",
)

messages = [
    {"role": "user", "content": "What is a GPU?"}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(
    text,
    return_tensors="pt",
).to(model.device)

with torch.no_grad():
    outputs = model(**inputs)

print("Input shape:", inputs["input_ids"].shape)
print("Logits shape:", outputs.logits.shape)