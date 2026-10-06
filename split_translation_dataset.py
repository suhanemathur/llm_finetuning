import json
import random

INPUT_FILE = "translation_train.jsonl"

TRAIN_FILE = "translation_train_split.jsonl"
EVAL_FILE = "translation_eval.jsonl"

SEED = 42
EVAL_SIZE = 5000

print("Loading dataset...")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    examples = [json.loads(line) for line in f]

print("Total examples:", len(examples))

random.seed(SEED)
random.shuffle(examples)

eval_examples = examples[:EVAL_SIZE]
train_examples = examples[EVAL_SIZE:]

print("Training examples:", len(train_examples))
print("Evaluation examples:", len(eval_examples))

with open(TRAIN_FILE, "w", encoding="utf-8") as f:
    for example in train_examples:
        f.write(json.dumps(example, ensure_ascii=False) + "\n")

with open(EVAL_FILE, "w", encoding="utf-8") as f:
    for example in eval_examples:
        f.write(json.dumps(example, ensure_ascii=False) + "\n")

print()
print("Saved:")
print("  ", TRAIN_FILE)
print("  ", EVAL_FILE)