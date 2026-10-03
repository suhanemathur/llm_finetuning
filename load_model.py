import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="cuda",
)

print("Model loaded.")
print("Device:", model.device)

messages = [
    {
        "role": "user",
        "content": "Explain what a GPU is in one short paragraph."
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(text, return_tensors="pt").to(model.device)
print("Token IDs:", inputs["input_ids"][0].tolist())
print("Tokens:", tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))

print("\nInput tokens:", inputs["input_ids"].shape[1])

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
    )

new_tokens = outputs[0][inputs["input_ids"].shape[1]:]

answer = tokenizer.decode(
    new_tokens,
    skip_special_tokens=True,
)

print("\nMODEL RESPONSE:")
print(answer)