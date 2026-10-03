import gc
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_PATH = "qwen3-4b-lora"

QUESTIONS = [
    "What is a GPU?",
    "What is LoRA?",
    "What is an epoch?",
    "Explain gradient accumulation in simple terms.",
]


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def generate_answer(model, question):
    messages = [
        {
            "role": "user",
            "content": question,
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    return tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    )


print("Loading original model...")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="cuda",
)

base_model.eval()

base_answers = {}

for question in QUESTIONS:
    base_answers[question] = generate_answer(
        base_model,
        question,
    )

del base_model
gc.collect()
torch.cuda.empty_cache()


print("\nLoading fine-tuned model...")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="cuda",
)

fine_tuned_model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
)

fine_tuned_model.eval()


for question in QUESTIONS:
    answer = generate_answer(
        fine_tuned_model,
        question,
    )

    print("\n" + "=" * 70)
    print("QUESTION:", question)

    print("\nORIGINAL MODEL:")
    print(base_answers[question])

    print("\nFINE-TUNED MODEL:")
    print(answer)


print("\nComparison complete.")