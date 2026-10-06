import json
from datasets import load_dataset

DATASET_NAME = "ai4bharat/indic-instruct-data-v0.1"
CONFIG_NAME = "nmt-seed"
SPLIT = "hi"

OUTPUT_FILE = "translation_train.jsonl"

print("Loading dataset...")

dataset = load_dataset(
    DATASET_NAME,
    CONFIG_NAME,
    split=SPLIT,
)

print("Total examples:", len(dataset))

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    for example in dataset:
        input_text = example["input_text"].strip()
        output_text = example["output_text"].strip()

        messages = [
            {
                "role": "user",
                "content": (
                    "Translate the following English sentence into Hindi:\n\n"
                    + input_text
                ),
            },
            {
                "role": "assistant",
                "content": output_text,
            },
        ]

        record = {
            "messages": messages
        }

        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print("Saved:", OUTPUT_FILE)