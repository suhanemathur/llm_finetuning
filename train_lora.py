import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer


MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DATASET_PATH = "training_data.jsonl"
OUTPUT_DIR = "qwen3-4b-lora"


print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="cuda",
)


print("Loading dataset...")

dataset = load_dataset(
    "json",
    data_files=DATASET_PATH,
    split="train",
)


print("Dataset size:", len(dataset))


lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


training_args = SFTConfig(
    output_dir=OUTPUT_DIR,

    num_train_epochs=3,

    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,

    learning_rate=2e-4,

    logging_steps=1,

    save_strategy="epoch",

    bf16=True,

    assistant_only_loss=True,

    report_to="none",
)


trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=lora_config,
)


print("\nStarting training...\n")

trainer.train()

print("\nTraining complete.")

trainer.save_model(OUTPUT_DIR)

print("Adapter saved to:", OUTPUT_DIR)