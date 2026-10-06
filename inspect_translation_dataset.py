import json
from transformers import AutoTokenizer

DATASET_FILE = "translation_train.jsonl"
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

token_lengths = []

with open(DATASET_FILE, "r", encoding="utf-8") as f:
    for line in f:
        example = json.loads(line)
        messages = example["messages"]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        tokens = tokenizer(
            text,
            add_special_tokens=False,
        )["input_ids"]

        token_lengths.append(len(tokens))

token_lengths.sort()

total = len(token_lengths)

def percentile(values, p):
    index = int((p / 100) * (len(values) - 1))
    return values[index]

print()
print("Total examples:", total)
print("Minimum tokens:", min(token_lengths))
print("25th percentile:", percentile(token_lengths, 25))
print("Median:", percentile(token_lengths, 50))
print("75th percentile:", percentile(token_lengths, 75))
print("90th percentile:", percentile(token_lengths, 90))
print("95th percentile:", percentile(token_lengths, 95))
print("99th percentile:", percentile(token_lengths, 99))
print("Maximum tokens:", max(token_lengths))
print("Average tokens:", sum(token_lengths) / total)

print()
print("Examples above 256 tokens:",
      sum(x > 256 for x in token_lengths))

print("Examples above 512 tokens:",
      sum(x > 512 for x in token_lengths))

print("Examples above 768 tokens:",
      sum(x > 768 for x in token_lengths))