from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

dataset = load_dataset(
    "json",
    data_files="training_data.jsonl",
    split="train",
)

example = dataset[0]

print("MESSAGES:")
print(example["messages"])

text = tokenizer.apply_chat_template(
    example["messages"],
    tokenize=False,
    add_generation_prompt=False,
)

print("\nCHAT TEMPLATE:")
print(text)

tokens = tokenizer(
    text,
    add_special_tokens=False,
)["input_ids"]

print("\nTOKEN IDs:")
print(tokens)

print("\nTOKENS:")
print(tokenizer.convert_ids_to_tokens(tokens))

print("\nNUMBER OF TOKENS:")
print(len(tokens))