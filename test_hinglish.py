import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_PATH = "qwen3-4b-hindi-translation"

TEST_MESSAGES = [
    "आज meeting बहुत important है, इसलिए हमें समय पर पहुंचना चाहिए।",
    "मुझे इस project की final report आज submit करनी है।",
    "अगर weather खराब हुआ तो flight को postpone करना पड़ेगा।",
    "इस sensor का temperature अभी बहुत high है।",
    "Please मुझे यह data Excel file में भेज दो।",
    "हमने model को तीन epochs तक train किया और फिर evaluate किया।",
    "इस problem को solve करने के लिए हमें पहले root cause identify करना होगा।",
    "कल team के साथ एक technical discussion schedule है।",
    "अगर GPU memory full हो गई तो training automatically stop हो सकती है।",
    "यह approach पहले वाले method से ज्यादा efficient है।",
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

for i, message in enumerate(TEST_MESSAGES, start=1):
    response = generate_response(
        base_model,
        tokenizer,
        message,
    )

    base_results.append(response)

    print(f"{i}. Input:")
    print(message)
    print("Base Qwen:")
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

    print(f"{i}. Input:")
    print(message)
    print("Base Qwen:")
    print(base_results[i - 1])
    print("Fine-tuned:")
    print(response)
    print("-" * 80)
