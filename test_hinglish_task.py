import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_PATH = "qwen3-4b-hindi-translation"

TEST_MESSAGES = [
    "Rewrite this sentence naturally in Hinglish:\n\nThe meeting is very important, so we should arrive on time.",

    "Rewrite this sentence naturally in Hinglish:\n\nI need to submit the final report for this project today.",

    "Rewrite this sentence naturally in Hinglish:\n\nIf the weather is bad, we will have to postpone the flight.",

    "Rewrite this sentence naturally in Hinglish:\n\nThe temperature of this sensor is very high right now.",

    "Rewrite this sentence naturally in Hinglish:\n\nWe trained the model for three epochs and then evaluated it.",

    "Explain gradient accumulation in simple Hinglish.",

    "Explain what GPU memory means in simple Hinglish.",

    "Explain LoRA fine-tuning in simple Hinglish.",

    "Explain why training data quality is important in simple Hinglish.",

    "Explain what a learning rate does in simple Hinglish.",
]


def generate_response(model, tokenizer, message):
    messages = [
        {
            "role": "user",
            "content": message,
        }
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
        outputs = model.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=False,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    return tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()


print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="cuda",
)

print("\nTesting BASE MODEL...\n")

base_results = []

for i, message in enumerate(TEST_MESSAGES, start=1):
    response = generate_response(
        base_model,
        tokenizer,
        message,
    )

    base_results.append(response)

    print(f"{i}. Instruction:")
    print(message)
    print("\nBase Qwen:")
    print(response)
    print("-" * 80)


print("\nLoading LoRA adapter...")

lora_model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
)

lora_model.eval()

print("\nTesting ENGLISH→HINDI FINE-TUNED MODEL...\n")

for i, message in enumerate(TEST_MESSAGES, start=1):
    response = generate_response(
        lora_model,
        tokenizer,
        message,
    )

    print(f"{i}. Instruction:")
    print(message)
    print("\nBase Qwen:")
    print(base_results[i - 1])
    print("\nFine-tuned:")
    print(response)
    print("-" * 80)
