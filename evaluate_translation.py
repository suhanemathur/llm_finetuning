import math
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_PATH = "qwen3-4b-hindi-translation"
EVAL_FILE = "translation_eval.jsonl"

BATCH_SIZE = 8
MAX_LENGTH = 256


def prepare_example(example, tokenizer):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"][0]

    # ---------------------------------------------------------
    # Only calculate loss on assistant tokens.
    # ---------------------------------------------------------

    assistant_start = text.find("<|im_start|>assistant")

    if assistant_start == -1:
        raise ValueError("Assistant section not found.")

    prefix = text[:assistant_start]

    prefix_tokens = tokenizer(
        prefix,
        truncation=True,
        max_length=MAX_LENGTH,
        add_special_tokens=False,
    )["input_ids"]

    assistant_start_token = len(prefix_tokens)

    labels = input_ids.clone()

    labels[:assistant_start_token] = -100

    return input_ids, labels


def evaluate_model(model, tokenizer, dataset, name):

    print("\n" + "=" * 70)
    print(f"Evaluating: {name}")
    print("=" * 70)

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    batch_input_ids = []
    batch_labels = []

    for index, example in enumerate(dataset):

        input_ids, labels = prepare_example(
            example,
            tokenizer,
        )

        batch_input_ids.append(input_ids)
        batch_labels.append(labels)

        if (
            len(batch_input_ids) == BATCH_SIZE
            or index == len(dataset) - 1
        ):

            max_len = max(
                x.shape[0]
                for x in batch_input_ids
            )

            padded_inputs = []
            padded_labels = []

            for ids, lbls in zip(
                batch_input_ids,
                batch_labels,
            ):

                padding = max_len - ids.shape[0]

                padded_inputs.append(
                    torch.cat(
                        [
                            ids,
                            torch.full(
                                (padding,),
                                tokenizer.pad_token_id,
                                dtype=ids.dtype,
                            ),
                        ]
                    )
                )

                padded_labels.append(
                    torch.cat(
                        [
                            lbls,
                            torch.full(
                                (padding,),
                                -100,
                                dtype=lbls.dtype,
                            ),
                        ]
                    )
                )

            input_ids = torch.stack(
                padded_inputs
            ).to(model.device)

            labels = torch.stack(
                padded_labels
            ).to(model.device)

            attention_mask = (
                input_ids
                != tokenizer.pad_token_id
            )

            with torch.no_grad():

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                logits = outputs.logits

            # Causal LM:
            # token N predicts token N+1

            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]

            predictions = shift_logits.argmax(dim=-1)

            valid = shift_labels != -100

            correct = (
                predictions[valid]
                == shift_labels[valid]
            ).sum().item()

            token_count = valid.sum().item()

            total_correct += correct
            total_tokens += token_count

            # Cross entropy over assistant tokens
            loss = torch.nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )

            total_loss += loss.item()

            batch_input_ids = []
            batch_labels = []

            if (index + 1) % 500 == 0:
                current_loss = (
                    total_loss / total_tokens
                )

                current_accuracy = (
                    total_correct / total_tokens
                )

                print(
                    f"{index + 1}/{len(dataset)} | "
                    f"loss={current_loss:.4f} | "
                    f"accuracy={current_accuracy:.4f}"
                )

    final_loss = total_loss / total_tokens
    final_accuracy = total_correct / total_tokens
    perplexity = math.exp(final_loss)

    print("\nResults:")
    print(f"Eval loss:       {final_loss:.6f}")
    print(f"Token accuracy:  {final_accuracy:.6f}")
    print(f"Perplexity:      {perplexity:.6f}")

    return {
        "loss": final_loss,
        "accuracy": final_accuracy,
        "perplexity": perplexity,
    }


print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token


print("Loading evaluation dataset...")

dataset = load_dataset(
    "json",
    data_files=EVAL_FILE,
    split="train",
)

print("Evaluation examples:", len(dataset))


# ============================================================
# BASE MODEL
# ============================================================

print("\nLoading base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="cuda",
)

base_results = evaluate_model(
    base_model,
    tokenizer,
    dataset,
    "BASE MODEL",
)

del base_model
torch.cuda.empty_cache()


# ============================================================
# HINDI LORA
# ============================================================

print("\nLoading base model for LoRA...")

adapter_base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="cuda",
)

print("Loading Hindi LoRA adapter...")

lora_model = PeftModel.from_pretrained(
    adapter_base,
    ADAPTER_PATH,
)

lora_results = evaluate_model(
    lora_model,
    tokenizer,
    dataset,
    "HINDI LORA MODEL",
)


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

print(
    f"Base loss:       "
    f"{base_results['loss']:.6f}"
)

print(
    f"LoRA loss:       "
    f"{lora_results['loss']:.6f}"
)

loss_change = (
    lora_results["loss"]
    - base_results["loss"]
)

loss_improvement = (
    (base_results["loss"] - lora_results["loss"])
    / base_results["loss"]
) * 100

print(
    f"\nLoss change:     "
    f"{loss_change:+.6f}"
)

print(
    f"Loss improvement:"
    f" {loss_improvement:+.2f}%"
)

print(
    f"\nBase accuracy:   "
    f"{base_results['accuracy']:.6f}"
)

print(
    f"LoRA accuracy:   "
    f"{lora_results['accuracy']:.6f}"
)

accuracy_change = (
    lora_results["accuracy"]
    - base_results["accuracy"]
)

print(
    f"Accuracy change:"
    f" {accuracy_change:+.6f}"
)

print(
    f"\nBase perplexity: "
    f"{base_results['perplexity']:.6f}"
)

print(
    f"LoRA perplexity: "
    f"{lora_results['perplexity']:.6f}"
)

print("\nEvaluation complete.")