import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer


MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

TRAIN_FILE = "translation_train_split.jsonl"
EVAL_FILE = "translation_eval.jsonl"

OUTPUT_DIR = "qwen3-4b-hindi-translation"


print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="cuda",
)


print("Loading training dataset...")

train_dataset = load_dataset(
    "json",
    data_files=TRAIN_FILE,
    split="train",
)

print("Training examples:", len(train_dataset))


print("Loading evaluation dataset...")

eval_dataset = load_dataset(
    "json",
    data_files=EVAL_FILE,
    split="train",
)

print("Evaluation examples:", len(eval_dataset))


lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


training_args = SFTConfig(
    output_dir=OUTPUT_DIR,

    num_train_epochs=1,

    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,

    learning_rate=2e-4,

    max_length=512,
    packing=True,

    logging_steps=50,
    save_strategy="epoch",
    eval_strategy="steps",
    eval_steps=500,

    bf16=True,

    assistant_only_loss=True,

    report_to="none",
)


trainer = SFTTrainer(
    model=model,
    args=training_args,

    train_dataset=train_dataset,
    eval_dataset=eval_dataset,

    processing_class=tokenizer,

    peft_config=lora_config,
)


print("\nStarting training...\n")

trainer.train()


print("\nTraining complete.")

trainer.save_model(OUTPUT_DIR)

print("Adapter saved to:", OUTPUT_DIR)