import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_PATH = "qwen3-4b-hindi-translation"

TEST_SENTENCES = [
    "The weather is beautiful today.",
    "The company announced that it will launch a new product next year.",
    "Please send me the report before the meeting starts.",
    "The aircraft completed the mission safely despite strong winds.",
    "Machine learning models require high-quality training data.",
    "She has been working on this project for three months.",
    "We need to reduce the processing time without affecting accuracy.",
    "The engineer inspected the system and found a damaged sensor.",
    "If the temperature increases further, the system may shut down automatically.",
    "The team decided to postpone the experiment until more data was available.",
]


def translate(model, tokenizer, sentence):
    messages = [
        {
            "role": "user",
            "content": (
                "Translate the following English sentence into Hindi:\n\n"
                + sentence
            ),
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
            max_new_tokens=150,
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

for i, sentence in enumerate(TEST_SENTENCES, start=1):
    translation = translate(base_model, tokenizer, sentence)

    base_results.append(translation)

    print(f"{i}. English:")
    print(sentence)
    print("Base Qwen:")
    print(translation)
    print("-" * 80)


print("\nLoading LoRA adapter...")

lora_model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
)

lora_model.eval()

print("\nTesting FINE-TUNED MODEL...\n")

for i, sentence in enumerate(TEST_SENTENCES, start=1):
    translation = translate(lora_model, tokenizer, sentence)

    print(f"{i}. English:")
    print(sentence)
    print("Base Qwen:")
    print(base_results[i - 1])
    print("Fine-tuned:")
    print(translation)
    print("-" * 80)
