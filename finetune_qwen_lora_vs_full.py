#!/usr/bin/env python3
"""Compare LoRA ranks and full-weight SFT for Qwen2.5-0.5B.

Install the required packages once:

    pip install datasets peft accelerate matplotlib

Then run:

    python finetune_qwen_lora_vs_full.py

The script trains four fresh copies of the same base checkpoint, writes raw
loss histories and summary measurements, plots training/validation loss, and
saves held-out generations from the base model and every fine-tuned model.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Callable, Iterable


# ---------------------------------------------------------------------------
# Experiment settings. Edit these values to change the default experiment.
# ---------------------------------------------------------------------------

# Hugging Face checkpoint used as the common initialization for every run.
MODEL_NAME = "Qwen/Qwen2.5-0.5B"

# Multi-turn conversation dataset used for supervised fine-tuning.
DATASET_NAME = "HuggingFaceH4/ultrachat_200k"

# Dataset split containing supervised fine-tuning conversations.
TRAIN_SPLIT = "train_sft"

# Held-out dataset split used for validation and qualitative generation.
EVAL_SPLIT = "test_sft"

# Fixed number of shuffled training conversations selected from train_sft.
NUM_TRAIN_EXAMPLES = 1000

# Fixed number of shuffled held-out conversations used to calculate validation loss.
NUM_EVAL_EXAMPLES = 100

# Number of held-out user prompts used for qualitative comparisons.
NUM_QUALITATIVE_PROMPTS = 3

# LoRA ranks compared against full-weight fine-tuning.
LORA_RANKS = [1, 4, 16]

# Maximum tokenized conversation length used by every training configuration.
MAX_SEQUENCE_LENGTH = 512

# Number of complete passes through the fixed training subset.
NUM_EPOCHS = 1

# Number of conversations processed by each forward/backward pass.
MICRO_BATCH_SIZE = 1

# Number of micro-batches accumulated before each optimizer update.
GRADIENT_ACCUMULATION_STEPS = 8

# Effective batch size shared by all LoRA ranks and full-weight tuning.
EFFECTIVE_BATCH_SIZE = MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS

# Number of conversations processed together during validation.
EVAL_BATCH_SIZE = 2

# Learning rate shared by all LoRA ranks and full-weight tuning.
LEARNING_RATE = 5e-5

# AdamW weight decay shared by every fine-tuning configuration.
WEIGHT_DECAY = 0.01

# Fraction of optimizer steps used for linear learning-rate warmup.
WARMUP_RATIO = 0.05

# Maximum gradient norm used for clipping before optimizer updates.
MAX_GRAD_NORM = 1.0

# Number of optimizer updates between recorded training-loss points.
LOG_EVERY_STEPS = 10

# Number of optimizer updates between held-out validation evaluations.
EVAL_EVERY_STEPS = 25

# Dropout probability used by every LoRA adapter.
LORA_DROPOUT = 0.05

# Number of tokens generated for each qualitative held-out prompt.
GENERATION_MAX_NEW_TOKENS = 128

# Reproducibility seed used for subset selection, model initialization, and batches.
SEED = 2025

# Directory containing metrics, plots, generations, and optional checkpoints.
DEFAULT_OUTPUT_DIR = Path("results/qwen_lora_vs_full")

# Project-local Hugging Face cache used when HF_HOME is not already configured.
DEFAULT_HF_HOME = Path(__file__).resolve().parent / ".model_cache" / "huggingface"

# Project-local Matplotlib cache used when MPLCONFIGDIR is not already configured.
DEFAULT_MPL_CACHE = Path(__file__).resolve().parent / ".model_cache" / "matplotlib"

# Whether trained adapters and the full-weight model are saved after each run.
SAVE_CHECKPOINTS = True

# Minimum free GPU memory required before the experiment begins.
MINIMUM_FREE_GPU_GIB = 8.5

# Column order used by the per-configuration summary CSV.
SUMMARY_FIELDS = [
    "configuration",
    "tuning_method",
    "lora_rank",
    "trainable_parameters",
    "total_parameters",
    "trainable_percent",
    "peak_allocated_gib",
    "peak_reserved_gib",
    "gpu_total_gib",
    "peak_reserved_percent",
    "training_time_seconds",
    "final_training_loss",
    "final_validation_loss",
    "checkpoint_path",
]

# Column order used by the raw train/validation loss-history CSV.
LOSS_HISTORY_FIELDS = ["configuration", "split", "step", "epoch", "loss"]


# Parse output, subset, epoch, and checkpoint-saving command-line overrides.
def parse_args() -> argparse.Namespace:
    # Command-line parser for the fine-tuning experiment.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for metrics, plots, generations, and checkpoints",
    )
    parser.add_argument(
        "--train-examples",
        type=int,
        default=NUM_TRAIN_EXAMPLES,
        help="number of fixed train_sft conversations to use",
    )
    parser.add_argument(
        "--eval-examples",
        type=int,
        default=NUM_EVAL_EXAMPLES,
        help="number of fixed test_sft conversations used for validation",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help="number of training epochs for every configuration",
    )
    parser.add_argument(
        "--no-save-checkpoints",
        action="store_true",
        help="record metrics and generations without saving trained weights",
    )
    return parser.parse_args()


# Stop early with a useful message when a required Python package is absent.
def import_dependencies() -> dict[str, Any]:
    try:
        import torch
        from datasets import load_dataset
        import matplotlib.pyplot as plt
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing training dependency. Install the required packages with:\n"
            "  pip install datasets peft accelerate matplotlib\n"
            f"Original import error: {exc}"
        ) from exc

    return {
        "torch": torch,
        "load_dataset": load_dataset,
        "plt": plt,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
    }


# Verify that one sufficiently free NVIDIA GPU is available for all four runs.
def validate_gpu() -> None:
    # Non-invasive nvidia-smi query that does not initialize a CUDA context.
    query_command = [
        "nvidia-smi",
        "--query-gpu=name,memory.free,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        # Name and memory values for the first visible GPU.
        query_output = subprocess.check_output(
            query_command, text=True, stderr=subprocess.STDOUT
        ).splitlines()[0]
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError) as exc:
        raise SystemExit("No accessible NVIDIA GPU was found via nvidia-smi.") from exc

    # Parsed GPU name and free/total memory values in MiB.
    gpu_name, free_mib_text, total_mib_text = (
        field.strip() for field in query_output.split(",", maxsplit=2)
    )

    # Free GPU memory converted from MiB to GiB.
    free_gib = float(free_mib_text) / 1024

    # Total GPU memory converted from MiB to GiB.
    total_gib = float(total_mib_text) / 1024
    if free_gib < MINIMUM_FREE_GPU_GIB:
        raise SystemExit(
            f"Not enough free GPU memory: {free_gib:.2f}/{total_gib:.2f} GiB "
            f"is free on {gpu_name}, but at least {MINIMUM_FREE_GPU_GIB:.1f} GiB "
            "is required. Stop other GPU workloads (check nvidia-smi) and rerun."
        )
    print(f"Using {gpu_name}: {free_gib:.2f}/{total_gib:.2f} GiB free", flush=True)


# Seed Python and PyTorch before each run so batches and dropout are reproducible.
def seed_everything(torch: Any, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Render one UltraChat message sequence with Qwen's conversation markers.
def render_conversation(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    # Fallback Qwen ChatML rendering for a tokenizer without a bundled template.
    rendered_messages = [
        f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        for message in messages
    ]
    return "".join(rendered_messages)


# Render a held-out user message followed by an assistant generation marker.
def render_generation_prompt(tokenizer: Any, user_text: str) -> str:
    # Single-turn conversation used as the generation prefix.
    messages = [{"role": "user", "content": user_text}]
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"


# Tokenize the fixed conversation subset once for reuse by every configuration.
def tokenize_conversations(
    dataset: Any, tokenizer: Any, max_sequence_length: int
) -> list[dict[str, list[int]]]:
    # In-memory tokenized examples consumed by PyTorch DataLoaders.
    tokenized_examples: list[dict[str, list[int]]] = []
    for example in dataset:
        # Full multi-turn conversation rendered in the model's expected format.
        conversation_text = render_conversation(tokenizer, example["messages"])

        # Truncated token IDs and attention mask for this conversation.
        encoded = tokenizer(
            conversation_text,
            truncation=True,
            max_length=max_sequence_length,
            add_special_tokens=False,
        )
        if len(encoded["input_ids"]) >= 2:
            tokenized_examples.append(
                {
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                }
            )
    if not tokenized_examples:
        raise SystemExit("Tokenization produced no usable conversations.")
    return tokenized_examples


# Create a dynamic-padding collator that also constructs causal-language labels.
def make_collator(tokenizer: Any, torch: Any) -> Callable[[list[dict[str, Any]]], Any]:
    # Pad examples to the longest sequence in a batch and mask padding labels.
    def collate(examples: list[dict[str, Any]]) -> dict[str, Any]:
        # Dynamically padded input IDs and attention masks.
        batch = tokenizer.pad(examples, padding=True, return_tensors="pt")

        # Causal-LM targets equal to input IDs except at padding positions.
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch

    return collate


# Copy every tensor in one batch to the selected CUDA device.
def move_batch_to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {name: tensor.to(device, non_blocking=True) for name, tensor in batch.items()}


# Count all parameters and the subset that will receive gradient updates.
def count_parameters(model: Any) -> tuple[int, int]:
    # Total scalar parameters in the model and any attached adapters.
    total_parameters = sum(parameter.numel() for parameter in model.parameters())

    # Scalar parameters whose requires_grad flag is enabled.
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return trainable_parameters, total_parameters


# Evaluate token-weighted causal-language loss over the held-out DataLoader.
def evaluate_loss(model: Any, data_loader: Any, device: Any, torch: Any) -> float:
    model.eval()

    # Sum of per-token loss contributions across validation batches.
    weighted_loss_sum = 0.0

    # Number of non-padding target tokens across validation batches.
    target_token_count = 0
    with torch.inference_mode():
        for batch in data_loader:
            # Validation batch moved to the model's CUDA device.
            device_batch = move_batch_to_device(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Causal-language-model output containing the mean token loss.
                output = model(**device_batch)

            # Number of labels included in this batch's mean loss.
            batch_target_tokens = int((device_batch["labels"] != -100).sum().item())
            weighted_loss_sum += float(output.loss.item()) * batch_target_tokens
            target_token_count += batch_target_tokens
    model.train()
    return weighted_loss_sum / max(target_token_count, 1)


# Construct a linear-warmup, linear-decay multiplier for LambdaLR.
def learning_rate_multiplier(
    current_step: int, warmup_steps: int, total_steps: int
) -> float:
    if current_step < warmup_steps:
        return current_step / max(warmup_steps, 1)
    return max(
        0.0,
        (total_steps - current_step) / max(total_steps - warmup_steps, 1),
    )


# Train one model and record comparable train/validation loss histories.
def train_model(
    model: Any,
    train_loader: Any,
    eval_loader: Any,
    device: Any,
    torch: Any,
    configuration: str,
    epochs: int,
) -> tuple[list[dict[str, Any]], float]:
    # Trainable tensors supplied to AdamW.
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]

    # Shared optimizer configuration used by LoRA and full-weight tuning.
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Number of optimizer updates made during the complete training run.
    total_steps = math.ceil(len(train_loader) / GRADIENT_ACCUMULATION_STEPS) * epochs

    # Number of initial optimizer updates used for warmup.
    warmup_steps = math.ceil(total_steps * WARMUP_RATIO)

    # Shared linear warmup/decay learning-rate scheduler.
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: learning_rate_multiplier(step, warmup_steps, total_steps),
    )

    # Raw training and validation loss points returned for CSV and plotting.
    history: list[dict[str, Any]] = []

    # Initial held-out loss before any parameter updates.
    initial_validation_loss = evaluate_loss(model, eval_loader, device, torch)
    history.append(
        {
            "configuration": configuration,
            "split": "validation",
            "step": 0,
            "epoch": 0.0,
            "loss": initial_validation_loss,
        }
    )

    # Completed optimizer-update count shared across epochs.
    global_step = 0

    # Loss sum accumulated between training-history records.
    logging_loss_sum = 0.0

    # Micro-batch count accumulated between training-history records.
    logging_batch_count = 0
    optimizer.zero_grad(set_to_none=True)

    # Wall-clock timer including periodic validation but excluding model loading.
    training_start = time.perf_counter()
    model.train()
    for epoch_index in range(epochs):
        for batch_index, batch in enumerate(train_loader):
            # Training batch moved to the model's CUDA device.
            device_batch = move_batch_to_device(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Causal-language-model output for this training micro-batch.
                output = model(**device_batch)

                # Accumulation-scaled loss used for backpropagation.
                scaled_loss = output.loss / GRADIENT_ACCUMULATION_STEPS
            scaled_loss.backward()
            logging_loss_sum += float(output.loss.detach().item())
            logging_batch_count += 1

            # Whether this micro-batch completes an accumulation group or epoch.
            is_update_step = (
                (batch_index + 1) % GRADIENT_ACCUMULATION_STEPS == 0
                or batch_index + 1 == len(train_loader)
            )
            if not is_update_step:
                continue

            torch.nn.utils.clip_grad_norm_(trainable_parameters, MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            # Fractional epoch completed at this optimizer update.
            epoch_progress = epoch_index + (batch_index + 1) / len(train_loader)

            # Whether a training-loss record is due at this optimizer step.
            should_log = global_step % LOG_EVERY_STEPS == 0 or global_step == total_steps
            if should_log:
                # Mean micro-batch loss since the preceding training record.
                mean_training_loss = logging_loss_sum / max(logging_batch_count, 1)
                history.append(
                    {
                        "configuration": configuration,
                        "split": "training",
                        "step": global_step,
                        "epoch": epoch_progress,
                        "loss": mean_training_loss,
                    }
                )
                print(
                    f"  {configuration}: step {global_step}/{total_steps}, "
                    f"train loss={mean_training_loss:.4f}",
                    flush=True,
                )
                logging_loss_sum = 0.0
                logging_batch_count = 0

            # Whether held-out validation is due at this optimizer step.
            should_evaluate = (
                global_step % EVAL_EVERY_STEPS == 0 or global_step == total_steps
            )
            if should_evaluate:
                # Token-weighted held-out loss at the current checkpoint.
                validation_loss = evaluate_loss(model, eval_loader, device, torch)
                history.append(
                    {
                        "configuration": configuration,
                        "split": "validation",
                        "step": global_step,
                        "epoch": epoch_progress,
                        "loss": validation_loss,
                    }
                )
                print(
                    f"  {configuration}: step {global_step}/{total_steps}, "
                    f"validation loss={validation_loss:.4f}",
                    flush=True,
                )

    # Training and periodic-validation wall-clock duration.
    training_time_seconds = time.perf_counter() - training_start
    return history, training_time_seconds


# Extract fixed held-out prompts and reference replies for qualitative comparison.
def select_qualitative_examples(dataset: Any, count: int) -> list[dict[str, str]]:
    # Prompt/reference pairs selected from held-out multi-turn conversations.
    examples: list[dict[str, str]] = []
    for conversation in dataset:
        # Messages in the current held-out UltraChat conversation.
        messages = conversation["messages"]
        for message_index, message in enumerate(messages):
            if message["role"] != "user":
                continue

            # First assistant reply following this user message, when available.
            reference = ""
            for later_message in messages[message_index + 1 :]:
                if later_message["role"] == "assistant":
                    reference = later_message["content"]
                    break
            examples.append(
                {
                    "prompt": message["content"],
                    "held_out_reference": reference,
                }
            )
            break
        if len(examples) >= count:
            break
    if len(examples) < count:
        raise SystemExit("The held-out subset did not contain enough user prompts.")
    return examples


# Generate deterministic assistant replies for the shared held-out prompts.
def generate_replies(
    model: Any,
    tokenizer: Any,
    qualitative_examples: list[dict[str, str]],
    device: Any,
    torch: Any,
) -> list[str]:
    model.eval()

    # Generated assistant text aligned with qualitative_examples.
    replies: list[str] = []
    with torch.inference_mode():
        for example in qualitative_examples:
            # Chat-formatted single-user prompt for this example.
            prompt_text = render_generation_prompt(tokenizer, example["prompt"])

            # Tokenized prompt placed on the model's CUDA device.
            encoded_prompt = tokenizer(prompt_text, return_tensors="pt")
            encoded_prompt = move_batch_to_device(encoded_prompt, device)

            # Deterministic continuation from the current model.
            generated_ids = model.generate(
                **encoded_prompt,
                max_new_tokens=GENERATION_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # Generated token suffix excluding the original prompt tokens.
            continuation_ids = generated_ids[0, encoded_prompt["input_ids"].shape[1] :]
            replies.append(tokenizer.decode(continuation_ids, skip_special_tokens=True).strip())
    model.train()
    return replies


# Load a fresh base model and optionally attach an all-linear LoRA adapter.
def build_model(
    dependencies: dict[str, Any], rank: int | None, device: Any
) -> Any:
    # Imported PyTorch module used for the checkpoint dtype.
    torch = dependencies["torch"]

    # Fresh checkpoint ensures identical initialization across configurations.
    model = dependencies["AutoModelForCausalLM"].from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    if rank is not None:
        # PEFT configuration targeting every transformer linear layer.
        lora_config = dependencies["LoraConfig"](
            r=rank,
            lora_alpha=2 * rank,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )
        model = dependencies["get_peft_model"](model, lora_config)
        model.enable_input_require_grads()
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(True)

    return model.to(device)


# Write dictionaries to CSV using a fixed, reproducible column order.
def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        # CSV writer configured with the requested stable column order.
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# Plot training and validation loss in separate panels for all configurations.
def plot_loss_history(history: list[dict[str, Any]], output_dir: Path, plt: Any) -> None:
    # Two axes separating frequently logged training loss from validation loss.
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    # Stable plot colors for the three LoRA ranks and full-weight tuning.
    colors = {
        "LoRA r=1": "#2878B5",
        "LoRA r=4": "#D95319",
        "LoRA r=16": "#3A923A",
        "Full fine-tuning": "#7A4EAB",
    }
    for split, axis, title in (
        ("training", axes[0], "Training loss"),
        ("validation", axes[1], "Held-out validation loss"),
    ):
        for configuration, color in colors.items():
            # History points for one configuration and data split.
            points = [
                row
                for row in history
                if row["configuration"] == configuration and row["split"] == split
            ]
            if not points:
                continue
            axis.plot(
                [row["step"] for row in points],
                [row["loss"] for row in points],
                marker="o",
                markersize=3,
                linewidth=1.8,
                label=configuration,
                color=color,
            )
        axis.set_title(title)
        axis.set_xlabel("Optimizer step")
        axis.set_ylabel("Cross-entropy loss")
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False)

    for extension in ("png", "pdf"):
        figure.savefig(output_dir / f"loss_curves.{extension}", dpi=200)
    plt.close(figure)


# Release one trained model and its optimizer-related CUDA allocations.
def release_model(model: Any, torch: Any) -> None:
    model.to("cpu")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


# Run base generation, four comparable training jobs, and final reporting.
def main() -> None:
    # User-selected experiment overrides.
    args = parse_args()
    if args.train_examples < 1 or args.eval_examples < 1 or args.epochs < 1:
        raise SystemExit("Training examples, evaluation examples, and epochs must be positive.")

    DEFAULT_HF_HOME.mkdir(parents=True, exist_ok=True)
    DEFAULT_MPL_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))
    os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPL_CACHE))
    validate_gpu()

    # Imported training libraries made available after environment setup.
    dependencies = import_dependencies()

    # Imported PyTorch module used throughout the experiment.
    torch = dependencies["torch"]
    if not torch.cuda.is_available():
        raise SystemExit("PyTorch cannot access CUDA in this environment.")

    # CUDA device used for every model configuration.
    device = torch.device("cuda:0")
    seed_everything(torch, SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Shared tokenizer used to prepare identical examples for every model.
    tokenizer = dependencies["AutoTokenizer"].from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading fixed UltraChat subsets...", flush=True)

    # Complete training split before deterministic shuffle and selection.
    train_dataset = dependencies["load_dataset"](
        DATASET_NAME, split=TRAIN_SPLIT, cache_dir=str(DEFAULT_HF_HOME / "datasets")
    )
    if args.train_examples > len(train_dataset):
        raise SystemExit(
            f"Requested {args.train_examples} training examples, but only "
            f"{len(train_dataset)} are available."
        )
    train_dataset = train_dataset.shuffle(seed=SEED).select(range(args.train_examples))

    # Complete held-out split before deterministic shuffle and selection.
    eval_dataset = dependencies["load_dataset"](
        DATASET_NAME, split=EVAL_SPLIT, cache_dir=str(DEFAULT_HF_HOME / "datasets")
    )
    if args.eval_examples > len(eval_dataset):
        raise SystemExit(
            f"Requested {args.eval_examples} evaluation examples, but only "
            f"{len(eval_dataset)} are available."
        )
    eval_dataset = eval_dataset.shuffle(seed=SEED + 1).select(range(args.eval_examples))

    # Tokenized training examples reused without modification across all runs.
    tokenized_train = tokenize_conversations(
        train_dataset, tokenizer, MAX_SEQUENCE_LENGTH
    )

    # Tokenized held-out examples reused without modification across all runs.
    tokenized_eval = tokenize_conversations(eval_dataset, tokenizer, MAX_SEQUENCE_LENGTH)

    # Shared dynamic-padding batch collator.
    collator = make_collator(tokenizer, torch)

    # Fixed DataLoader order generator reset to this seed before each run.
    loader_seed = SEED + 2

    # Held-out prompt/reference pairs used for every qualitative generation pass.
    qualitative_examples = select_qualitative_examples(
        eval_dataset, min(NUM_QUALITATIVE_PROMPTS, args.eval_examples)
    )

    # JSON-serializable qualitative report initialized with prompts and references.
    qualitative_report = [dict(example, generations={}) for example in qualitative_examples]

    print("Generating base-model responses...", flush=True)

    # Untuned base checkpoint used only for the qualitative baseline.
    base_model = build_model(dependencies, rank=None, device=device)
    base_model.config.use_cache = True

    # Base responses aligned with the held-out qualitative prompts.
    base_replies = generate_replies(
        base_model, tokenizer, qualitative_examples, device, torch
    )
    for report_entry, reply in zip(qualitative_report, base_replies):
        report_entry["generations"]["Base model"] = reply
    release_model(base_model, torch)
    del base_model

    # Four configurations represented as display label, method, and optional rank.
    configurations = [
        (f"LoRA r={rank}", "lora", rank) for rank in LORA_RANKS
    ]
    configurations.append(("Full fine-tuning", "full", None))

    # Per-configuration resource and final-loss records.
    summary_rows: list[dict[str, Any]] = []

    # Every raw training and validation loss point from all configurations.
    complete_history: list[dict[str, Any]] = []
    for configuration, tuning_method, rank in configurations:
        print(f"\nStarting {configuration}...", flush=True)
        seed_everything(torch, SEED)

        # Deterministically seeded generator controlling shuffled training batches.
        data_generator = torch.Generator()
        data_generator.manual_seed(loader_seed)

        # Training DataLoader rebuilt with the same shuffled order for every run.
        train_loader = torch.utils.data.DataLoader(
            tokenized_train,
            batch_size=MICRO_BATCH_SIZE,
            shuffle=True,
            generator=data_generator,
            collate_fn=collator,
            num_workers=0,
            pin_memory=True,
        )

        # Deterministic held-out DataLoader shared by every run.
        eval_loader = torch.utils.data.DataLoader(
            tokenized_eval,
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
            pin_memory=True,
        )

        # Fresh full or LoRA-wrapped model for this configuration.
        model = build_model(dependencies, rank=rank, device=device)

        # Trainable and total scalar parameter counts for reporting.
        trainable_count, total_count = count_parameters(model)
        print(
            f"  trainable parameters: {trainable_count:,}/{total_count:,} "
            f"({100 * trainable_count / total_count:.4f}%)",
            flush=True,
        )

        torch.cuda.reset_peak_memory_stats(device)

        # Loss history and timed duration for this training configuration.
        history, training_time_seconds = train_model(
            model,
            train_loader,
            eval_loader,
            device,
            torch,
            configuration,
            args.epochs,
        )
        complete_history.extend(history)

        # Peak live tensor allocation observed after model construction.
        peak_allocated_bytes = torch.cuda.max_memory_allocated(device)

        # Peak PyTorch CUDA reservation, used as peak GPU-memory utilization.
        peak_reserved_bytes = torch.cuda.max_memory_reserved(device)

        # Total usable CUDA memory on the selected GPU.
        gpu_total_bytes = torch.cuda.get_device_properties(device).total_memory

        # Final recorded training loss for this configuration.
        final_training_loss = [
            row["loss"] for row in history if row["split"] == "training"
        ][-1]

        # Final recorded validation loss for this configuration.
        final_validation_loss = [
            row["loss"] for row in history if row["split"] == "validation"
        ][-1]

        # Checkpoint directory recorded even when checkpoint saving is disabled.
        checkpoint_path = args.output_dir / "checkpoints" / configuration.lower().replace(
            " ", "_"
        ).replace("=", "")
        if SAVE_CHECKPOINTS and not args.no_save_checkpoints:
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(checkpoint_path, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint_path)

        model.config.use_cache = True

        # Fine-tuned responses aligned with the held-out qualitative prompts.
        fine_tuned_replies = generate_replies(
            model, tokenizer, qualitative_examples, device, torch
        )
        for report_entry, reply in zip(qualitative_report, fine_tuned_replies):
            report_entry["generations"][configuration] = reply

        summary_rows.append(
            {
                "configuration": configuration,
                "tuning_method": tuning_method,
                "lora_rank": "" if rank is None else rank,
                "trainable_parameters": trainable_count,
                "total_parameters": total_count,
                "trainable_percent": 100 * trainable_count / total_count,
                "peak_allocated_gib": peak_allocated_bytes / 1024**3,
                "peak_reserved_gib": peak_reserved_bytes / 1024**3,
                "gpu_total_gib": gpu_total_bytes / 1024**3,
                "peak_reserved_percent": 100 * peak_reserved_bytes / gpu_total_bytes,
                "training_time_seconds": training_time_seconds,
                "final_training_loss": final_training_loss,
                "final_validation_loss": final_validation_loss,
                "checkpoint_path": (
                    str(checkpoint_path)
                    if SAVE_CHECKPOINTS and not args.no_save_checkpoints
                    else ""
                ),
            }
        )

        # Persist partial results so completed configurations survive a later failure.
        write_csv(args.output_dir / "summary.csv", SUMMARY_FIELDS, summary_rows)
        write_csv(
            args.output_dir / "loss_history.csv",
            LOSS_HISTORY_FIELDS,
            complete_history,
        )
        (args.output_dir / "qualitative_generations.json").write_text(
            json.dumps(qualitative_report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        release_model(model, torch)
        del model

    # Complete experiment configuration saved beside the raw measurements.
    run_configuration = {
        "model_name": MODEL_NAME,
        "dataset_name": DATASET_NAME,
        "train_split": TRAIN_SPLIT,
        "eval_split": EVAL_SPLIT,
        "train_examples": args.train_examples,
        "eval_examples": args.eval_examples,
        "qualitative_prompts": len(qualitative_examples),
        "lora_ranks": LORA_RANKS,
        "lora_target_modules": "all-linear",
        "lora_alpha_rule": "2 * rank",
        "lora_dropout": LORA_DROPOUT,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "epochs": args.epochs,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "max_grad_norm": MAX_GRAD_NORM,
        "seed": SEED,
        "dtype": "bfloat16",
        "gradient_checkpointing": True,
        "sft_loss_scope": "all non-padding conversation tokens",
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_configuration, indent=2) + "\n", encoding="utf-8"
    )
    plot_loss_history(complete_history, args.output_dir, dependencies["plt"])

    print(f"\nSummary:        {args.output_dir / 'summary.csv'}")
    print(f"Loss history:   {args.output_dir / 'loss_history.csv'}")
    print(f"Loss plot:      {args.output_dir / 'loss_curves.png'}")
    print(f"Generations:    {args.output_dir / 'qualitative_generations.json'}")
    print(f"Configuration:  {args.output_dir / 'run_config.json'}")


if __name__ == "__main__":
    main()
