import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_PATH = "qwen3-4b-hindi-translation"

print("Loading base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="cuda",
)

print("Loading adapter...")

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
    is_trainable=True,
)

total_params = 0
total_abs = 0.0
total_sq = 0.0
max_abs = 0.0
tensor_count = 0

print("\nLoRA tensors:\n")

for name, param in model.named_parameters():

    if "lora_" not in name:
        continue

    tensor = param.detach().float()

    count = tensor.numel()
    abs_sum = tensor.abs().sum().item()
    sq_sum = (tensor ** 2).sum().item()
    tensor_max = tensor.abs().max().item()

    total_params += count
    total_abs += abs_sum
    total_sq += sq_sum
    max_abs = max(max_abs, tensor_max)
    tensor_count += 1

    print(
        f"{name:90s} "
        f"shape={tuple(param.shape)!s:20s} "
        f"params={count:,}"
    )

mean_abs = total_abs / total_params
l2_norm = total_sq ** 0.5

print("\n" + "=" * 80)
print("ADAPTER SUMMARY")
print("=" * 80)

print(f"LoRA tensors:       {tensor_count}")
print(f"LoRA parameters:    {total_params:,}")
print(f"Mean |weight|:      {mean_abs:.8f}")
print(f"L2 norm:            {l2_norm:.4f}")
print(f"Max |weight|:       {max_abs:.8f}")