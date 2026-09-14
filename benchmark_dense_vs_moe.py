#!/usr/bin/env python3
"""Benchmark dense and MoE checkpoints with one GPU and vLLM.

Edit PREFILL_PROMPT and DECODE_PROMPT below, then run:

    python benchmark_dense_vs_moe.py

The script writes one raw CSV and two PNG/PDF figures under ``results/``.
Each model is run in a fresh subprocess so its vLLM engine fully releases the
GPU before the next checkpoint is loaded.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# Experiment settings. These are the two prompt variables to edit.
# ---------------------------------------------------------------------------

# Base text whose token IDs are repeated to construct each prefill input length.
PREFILL_PROMPT = (
   "In a hole in the ground there lived a hobbit."
   "Not a nasty, dirty, wet hole, filled with the ends of worms and an oozy smell,"
   "nor yet a dry, bare, sandy hole with nothing in it to sit down on or eat: "
   "it was a hobbit-hole, and that means comfort."
   "What book is this the opening paragraph of?"
)

# Short input used for every decode-dominated generation request.
DECODE_PROMPT = "Summarize the story of the hobbit in one sentence."

# Checkpoint IDs and readable labels used when running and plotting the models.
MODELS = [
    ("allenai/OLMo-2-0425-1B-Instruct", "Dense 1B"),
    ("allenai/OLMoE-1B-7B-0924-Instruct", "MoE 1B/7B"),
    ("allenai/OLMo-2-1124-7B", "Dense 7B"),
]

# Input-token counts evaluated by the prefill-dominated benchmark.
PREFILL_LENGTHS = [128, 256, 512, 768, 1024]

# Number of requests included in each prefill measurement.
PREFILL_BATCH_SIZE = 1

# Numbers of simultaneous requests evaluated by the decode benchmark.
DECODE_CONCURRENCIES = [1, 2, 4, 8, 16, 32]

# Maximum requests resident in vLLM at once. A 32-request benchmark therefore
# also measures saturation and queueing beyond the server's resident capacity.
MAX_NUM_SEQS = 16

# Number of tokens every decode request must generate.
DECODE_OUTPUT_TOKENS = 256

# Number of timed samples collected for every plotted point.
REPEATS = 3

# Number of untimed runs performed before timing each configuration.
WARMUP_RUNS = 1

# Maximum combined prompt and generated length accepted by the vLLM engine.
MAX_MODEL_LEN = max(PREFILL_LENGTHS) + 1

# Maximum tokens vLLM processes in one scheduler iteration. Longer prefills are
# chunked; this cap leaves enough activation memory for Dense 7B on a 10 GB GPU.
MAX_NUM_BATCHED_TOKENS = 1024

# Fraction of GPU memory vLLM may reserve when no fixed cache size is supplied.
# The explicit KV_CACHE_MEMORY_BYTES setting below takes precedence in this run.
GPU_MEMORY_UTILIZATION = 0.96

# Fixed FP8 KV-cache allocation shared by every model. This is large enough for
# the longest individual request while leaving Dense 7B runtime workspace.
KV_CACHE_MEMORY_BYTES = 288 * 1024**2

# Random seed passed to every model engine for reproducibility.
SEED = 17

# Writable project-local fallback for Hugging Face model downloads. Set HF_HOME
# before launching the script if you prefer a different shared cache location.
MODEL_CACHE_DIR = Path(__file__).resolve().parent / ".model_cache" / "huggingface"

# Writable project-local cache used by Matplotlib for font metadata.
MATPLOTLIB_CACHE_DIR = Path(__file__).resolve().parent / ".model_cache" / "matplotlib"

# Weight-quantization mode passed to vLLM for every checkpoint.
WEIGHT_QUANTIZATION = "fp8"

# Data type used to store attention keys and values during generation.
KV_CACHE_DTYPE = "fp8"

# Attention implementation used consistently across all three checkpoints.
ATTENTION_BACKEND = "FLASHINFER"

# Disable the incompatible Inductor model compilation pass and CUDA graphs.
# Eager execution saves the memory needed to fit Dense 7B on a 10 GB GPU.
COMPILATION_CONFIG = {
    "mode": 0,
    "cudagraph_mode": "NONE",
}

# Ordered column names used when writing and reading raw measurements.
CSV_FIELDS = [
    "model",
    "label",
    "architecture",
    "workload",
    "repeat",
    "requested_input_tokens",
    "input_tokens",
    "parallel_generations",
    "requested_output_tokens_per_request",
    "output_tokens",
    "elapsed_seconds",
    "tokens_per_second",
    "weight_quantization",
    "kv_cache_dtype",
]


# Parse public command-line options and private subprocess-worker options.
def parse_args() -> argparse.Namespace:
    # Parser that defines and validates the command-line interface.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for raw data and plots (default: results)",
    )
    parser.add_argument(
        "--repeats", type=int, default=REPEATS, help="Timed repetitions per point"
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional model labels or Hugging Face IDs to run",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Recreate plots from the existing raw CSV without loading models",
    )
    # Private arguments used by the clean subprocess launched for each model.
    parser.add_argument("--_worker-model", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-label", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_raw-csv", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


# Convert a model's readable label into its dense or MoE architecture category.
def architecture(label: str) -> str:
    return "moe" if label.startswith("MoE") else "dense"


# Verify that CUDA is visible and enough GPU memory is free for vLLM's request.
def validate_gpu() -> None:
    # Read-only GPU query that does not create a memory-consuming CUDA context.
    query_command = [
        "nvidia-smi",
        "--query-gpu=name,compute_cap,memory.free,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        # First GPU's comma-separated name, capability, and MiB memory values.
        query_output = subprocess.check_output(
            query_command, text=True, stderr=subprocess.STDOUT
        ).splitlines()[0]
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError) as exc:
        raise SystemExit(
            "No NVIDIA GPU is available. Check that nvidia-smi works and that "
            "this process has access to the GPU."
        ) from exc

    # Parsed nvidia-smi fields for the selected first GPU.
    gpu_name, capability_text, free_mib_text, total_mib_text = (
        field.strip() for field in query_output.split(",", maxsplit=3)
    )

    # Currently free and total bytes reported by nvidia-smi.
    free_bytes = float(free_mib_text) * 1024**2
    total_bytes = float(total_mib_text) * 1024**2

    # Bytes vLLM will require to satisfy GPU_MEMORY_UTILIZATION.
    requested_bytes = total_bytes * GPU_MEMORY_UTILIZATION

    # Approximate memory reserved by the display driver but excluded from the
    # CUDA total that vLLM uses internally.
    driver_reserve_bytes = 512 * 1024**2
    if free_bytes + driver_reserve_bytes < requested_bytes:
        # Byte-to-GiB divisor used to make the error message easy to read.
        gib = 1024**3
        raise SystemExit(
            "Not enough free GPU memory to start vLLM: "
            f"{free_bytes / gib:.2f} GiB free, but "
            f"GPU_MEMORY_UTILIZATION={GPU_MEMORY_UTILIZATION} requests "
            f"{requested_bytes / gib:.2f} GiB of the {total_bytes / gib:.2f} GiB GPU. "
            "Stop other GPU workloads (check with nvidia-smi), then rerun."
        )

    # Compute capability used to select a compatible FP8 matrix kernel.
    capability = tuple(int(part) for part in capability_text.split(".", maxsplit=1))
    if capability < (8, 9) and WEIGHT_QUANTIZATION == "fp8":
        # Native FP8 CUTLASS fails on pre-Ada GPUs; vLLM's Marlin fallback keeps
        # weights in FP8 while multiplying them with 16-bit activations.
        disabled_kernels = os.environ.get("VLLM_DISABLED_KERNELS", "").split(",")
        if "CutlassFP8ScaledMMLinearKernel" not in disabled_kernels:
            disabled_kernels.append("CutlassFP8ScaledMMLinearKernel")
        os.environ["VLLM_DISABLED_KERNELS"] = ",".join(
            kernel for kernel in disabled_kernels if kernel
        )

    print(
        f"Using {gpu_name} "
        f"(compute capability {capability[0]}.{capability[1]}, "
        f"{free_bytes / 1024**3:.2f} GiB free)",
        flush=True,
    )


# Build a prompt containing exactly the requested number of token IDs.
def repeated_token_ids(tokenizer: Any, text: str, target_length: int) -> list[int]:
    """Tokenize text, then repeat/truncate its tokens to an exact length."""
    # Token IDs for one copy of the user-configured prefill text.
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError("PREFILL_PROMPT must produce at least one token")
    # Number of copies needed before truncating to the exact target length.
    copies = math.ceil(target_length / len(ids))
    return (ids * copies)[:target_length]


# Wait for queued CUDA work so wall-clock timing includes all GPU computation.
def cuda_sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


# Run one synchronous vLLM batch and return its outputs and elapsed seconds.
def timed_generate(llm: Any, prompts: list[Any], sampling_params: Any) -> tuple[Any, float]:
    cuda_sync()
    # High-resolution timestamp immediately before generation starts.
    start = time.perf_counter()
    # Completed vLLM request outputs for every prompt in the submitted batch.
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    cuda_sync()
    return outputs, time.perf_counter() - start


# Append and immediately flush one measurement so partial runs remain usable.
def append_row(csv_path: Path, row: dict[str, Any]) -> None:
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        # CSV serializer configured with the shared, stable column order.
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


# Load one model, benchmark both workloads, and save every raw observation.
def run_worker(model: str, label: str, csv_path: Path, repeats: int) -> None:
    validate_gpu()

    # Imports stay in the child process. This is important: initializing CUDA in
    # the parent can prevent vLLM from cleanly starting successive engines.
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise SystemExit(
            "vLLM is not installed. Install a GPU-compatible vLLM build first "
            "(for example: pip install vllm)."
        ) from exc

    print(f"\nLoading {label}: {model}", flush=True)
    try:
        # Single-GPU vLLM engine configured identically for every checkpoint.
        llm = LLM(
            model=model,
            quantization=WEIGHT_QUANTIZATION,
            kv_cache_dtype=KV_CACHE_DTYPE,
            attention_backend=ATTENTION_BACKEND,
            tensor_parallel_size=1,
            max_model_len=MAX_MODEL_LEN,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            max_num_seqs=MAX_NUM_SEQS,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            kv_cache_memory_bytes=KV_CACHE_MEMORY_BYTES,
            enable_prefix_caching=False,
            compilation_config=COMPILATION_CONFIG,
            seed=SEED,
            trust_remote_code=True,
        )
    except ValueError as exc:
        if "Free memory on device" in str(exc):
            raise SystemExit(
                "GPU memory became unavailable while vLLM was starting. Stop "
                "other GPU services (check nvidia-smi or Docker), then rerun. "
                f"vLLM reported: {exc}"
            ) from None
        raise
    # Model-specific tokenizer used to construct and count exact prompt tokens.
    tokenizer = llm.get_tokenizer()

    # Generation settings that isolate prefill by requesting only one new token.
    prefill_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        min_tokens=1,
        ignore_eos=True,
        detokenize=False,
    )
    # Generation settings that force a long, fixed-size decode for every request.
    decode_params = SamplingParams(
        temperature=0.0,
        max_tokens=DECODE_OUTPUT_TOKENS,
        min_tokens=DECODE_OUTPUT_TOKENS,
        ignore_eos=True,
        detokenize=False,
    )

    for requested_length in PREFILL_LENGTHS:
        if requested_length + 1 > MAX_MODEL_LEN:
            raise ValueError(
                f"Prefill length {requested_length} plus its output exceeds "
                f"MAX_MODEL_LEN={MAX_MODEL_LEN}"
            )
        # Exact-length token sequence used by this prefill configuration.
        token_ids = repeated_token_ids(tokenizer, PREFILL_PROMPT, requested_length)

        # One or more identical tokenized requests submitted as a single batch.
        prompts = [{"prompt_token_ids": token_ids} for _ in range(PREFILL_BATCH_SIZE)]
        for _ in range(WARMUP_RUNS):
            llm.generate(prompts, prefill_params, use_tqdm=False)
        for repeat in range(repeats):
            # Generated requests and wall-clock duration for this timed repetition.
            outputs, elapsed = timed_generate(llm, prompts, prefill_params)

            # Total number of prompt tokens processed across the batch.
            input_tokens = sum(len(output.prompt_token_ids) for output in outputs)

            # Total number of newly generated tokens across the batch.
            output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)

            # Complete raw measurement written as one CSV record.
            row = {
                "model": model,
                "label": label,
                "architecture": architecture(label),
                "workload": "prefill",
                "repeat": repeat + 1,
                "requested_input_tokens": requested_length,
                "input_tokens": input_tokens,
                "parallel_generations": PREFILL_BATCH_SIZE,
                "requested_output_tokens_per_request": 1,
                "output_tokens": output_tokens,
                "elapsed_seconds": elapsed,
                # Prefill throughput counts prompt tokens, not the one output token.
                "tokens_per_second": input_tokens / elapsed,
                "weight_quantization": WEIGHT_QUANTIZATION,
                "kv_cache_dtype": KV_CACHE_DTYPE,
            }
            append_row(csv_path, row)
            print(
                f"  prefill length={requested_length:4d} repeat={repeat + 1}: "
                f"{row['tokens_per_second']:.1f} input tok/s",
                flush=True,
            )

    # Tokenized short prompt shared by all decode requests for this model.
    decode_ids = tokenizer.encode(DECODE_PROMPT, add_special_tokens=False)
    if len(decode_ids) + DECODE_OUTPUT_TOKENS > MAX_MODEL_LEN:
        raise ValueError(
            "DECODE_PROMPT plus DECODE_OUTPUT_TOKENS exceeds MAX_MODEL_LEN: "
            f"{len(decode_ids)} + {DECODE_OUTPUT_TOKENS} > {MAX_MODEL_LEN}"
        )
    for concurrency in DECODE_CONCURRENCIES:
        # Batch containing the configured number of simultaneous decode requests.
        prompts = [{"prompt_token_ids": decode_ids} for _ in range(concurrency)]
        for _ in range(WARMUP_RUNS):
            llm.generate(prompts, decode_params, use_tqdm=False)
        for repeat in range(repeats):
            # Generated requests and wall-clock duration for this timed repetition.
            outputs, elapsed = timed_generate(llm, prompts, decode_params)

            # Total prompt tokens processed before decoding starts.
            input_tokens = sum(len(output.prompt_token_ids) for output in outputs)

            # Total newly generated tokens across all simultaneous requests.
            output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)

            # Complete raw measurement written as one CSV record.
            row = {
                "model": model,
                "label": label,
                "architecture": architecture(label),
                "workload": "decode",
                "repeat": repeat + 1,
                "requested_input_tokens": len(decode_ids),
                "input_tokens": input_tokens,
                "parallel_generations": concurrency,
                "requested_output_tokens_per_request": DECODE_OUTPUT_TOKENS,
                "output_tokens": output_tokens,
                "elapsed_seconds": elapsed,
                # Aggregate output throughput across all simultaneous requests.
                "tokens_per_second": output_tokens / elapsed,
                "weight_quantization": WEIGHT_QUANTIZATION,
                "kv_cache_dtype": KV_CACHE_DTYPE,
            }
            append_row(csv_path, row)
            print(
                f"  decode concurrency={concurrency:2d} repeat={repeat + 1}: "
                f"{row['tokens_per_second']:.1f} output tok/s",
                flush=True,
            )


# Load raw CSV records and convert numeric strings back to Python numbers.
def read_rows(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        # Raw measurement dictionaries loaded from all CSV records.
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in (
            "repeat",
            "requested_input_tokens",
            "input_tokens",
            "parallel_generations",
            "requested_output_tokens_per_request",
            "output_tokens",
        ):
            row[key] = int(row[key])
        for key in ("elapsed_seconds", "tokens_per_second"):
            row[key] = float(row[key])
    return rows


# Calculate a point's mean throughput and sample-standard-deviation error bar.
def mean_and_error(values: Iterable[float]) -> tuple[float, float]:
    # Materialized values so the iterable can be summarized more than once.
    samples = list(values)

    # Arithmetic mean displayed as the plotted point.
    mean = statistics.fmean(samples)

    # Sample standard deviation displayed as the error bar.
    error = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return mean, error


# Create the prefill and decode plots from the collected raw measurements.
def plot_results(csv_path: Path, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Plotting requires matplotlib: pip install matplotlib") from exc

    # Parsed raw data used to compute every plotted mean and error bar.
    rows = read_rows(csv_path)
    if not rows:
        raise SystemExit(f"No measurements found in {csv_path}")

    # Stable model-to-color mapping shared by both figures.
    colors = {"Dense 1B": "#2878B5", "MoE 1B/7B": "#D95319", "Dense 7B": "#3A923A"}

    # Plot definitions: workload, X field, labels, title, and output filename.
    specs = [
        (
            "prefill",
            "requested_input_tokens",
            "Input context length (tokens)",
            "Aggregate prefill throughput (input tokens/s)",
            "Prefill-dominated throughput",
            "prefill_throughput",
        ),
        (
            "decode",
            "parallel_generations",
            "Parallel generations",
            "Aggregate decode throughput (output tokens/s)",
            "Decode-dominated throughput",
            "decode_throughput",
        ),
    ]

    for workload, x_key, xlabel, ylabel, title, stem in specs:
        # Figure and axes for the current workload's plot.
        fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)

        # Labels actually present, allowing plots from a selected model subset.
        present_labels = {row["label"] for row in rows if row["workload"] == workload}
        for _, label in MODELS:
            if label not in present_labels:
                continue
            # All raw repetitions for one model and one workload.
            model_rows = [
                row for row in rows if row["workload"] == workload and row["label"] == label
            ]
            # Sorted context lengths or concurrency values used on the X axis.
            xs = sorted({row[x_key] for row in model_rows})

            # Mean throughputs and standard deviations aligned with each X value.
            means, errors = [], []
            for x in xs:
                # Summary statistics across the repetitions at this plotted point.
                mean, error = mean_and_error(
                    row["tokens_per_second"] for row in model_rows if row[x_key] == x
                )
                means.append(mean)
                errors.append(error)
            ax.errorbar(
                xs,
                means,
                yerr=errors,
                marker="o",
                linewidth=2,
                capsize=3,
                label=label,
                color=colors[label],
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        if workload == "decode":
            ax.set_xscale("log", base=2)
            ax.set_xticks(DECODE_CONCURRENCIES)
            ax.set_xticklabels([str(value) for value in DECODE_CONCURRENCIES])
        for extension in ("png", "pdf"):
            fig.savefig(output_dir / f"{stem}.{extension}", dpi=200)
        plt.close(fig)


# Resolve optional model IDs or labels into the configured model tuples.
def select_models(requested: list[str] | None) -> list[tuple[str, str]]:
    if not requested:
        return MODELS
    # Models whose ID or readable label was requested on the command line.
    selected = [pair for pair in MODELS if pair[0] in requested or pair[1] in requested]

    # Requested names that did not match any configured model.
    missing = set(requested) - {item for pair in selected for item in pair}
    if missing:
        # Human-readable list of valid values included in the error message.
        choices = ", ".join(f"{label!r} or {model!r}" for model, label in MODELS)
        raise SystemExit(f"Unknown --models value(s): {sorted(missing)}. Choices: {choices}")
    return selected


# Save the effective experiment configuration beside the measurements and plots.
def write_metadata(output_dir: Path, selected: list[tuple[str, str]], repeats: int) -> None:
    # Serializable snapshot of all settings needed to understand the run.
    metadata = {
        "models": [{"id": model, "label": label} for model, label in selected],
        "prefill_prompt": PREFILL_PROMPT,
        "decode_prompt": DECODE_PROMPT,
        "prefill_lengths": PREFILL_LENGTHS,
        "prefill_batch_size": PREFILL_BATCH_SIZE,
        "decode_concurrencies": DECODE_CONCURRENCIES,
        "decode_output_tokens": DECODE_OUTPUT_TOKENS,
        "repeats": repeats,
        "warmup_runs": WARMUP_RUNS,
        "max_model_len": MAX_MODEL_LEN,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "max_num_seqs": MAX_NUM_SEQS,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "kv_cache_memory_bytes": KV_CACHE_MEMORY_BYTES,
        "weight_quantization": WEIGHT_QUANTIZATION,
        "kv_cache_dtype": KV_CACHE_DTYPE,
        "attention_backend": ATTENTION_BACKEND,
        "compilation_config": COMPILATION_CONFIG,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


# Coordinate worker subprocesses, output-file creation, and final plotting.
def main() -> None:
    # Use the writable fallback only when the caller did not configure HF_HOME.
    os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR))

    # Ensure plotting works when the user's default Matplotlib config is read-only.
    MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))

    # Keep vLLM's engine core in this worker process so a second CUDA context
    # does not consume the memory margin needed by the Dense 7B checkpoint.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    # First available C compiler used by Triton to build its CUDA launcher.
    compiler = (
        shutil.which("cc")
        or shutil.which("gcc")
        or shutil.which("x86_64-conda-linux-gnu-cc")
    )
    if compiler:
        os.environ.setdefault("CC", compiler)
    elif not os.environ.get("CC"):
        raise SystemExit(
            "No C compiler was found. Triton requires one; install gcc or "
            "gcc_linux-64 before running this benchmark."
        )

    # First available C++ compiler used to link FlashInfer's generated kernel.
    cxx_compiler = (
        shutil.which("c++")
        or shutil.which("g++")
        or shutil.which("x86_64-conda-linux-gnu-c++")
    )
    if cxx_compiler:
        os.environ.setdefault("CXX", cxx_compiler)
    elif not os.environ.get("CXX"):
        raise SystemExit(
            "No C++ compiler was found. FlashInfer requires one; install g++ or "
            "gxx_linux-64 before running this benchmark."
        )

    # CUDA compiler used by FlashInfer to build its FP8 KV-cache kernel.
    nvcc = shutil.which("nvcc")
    if nvcc:
        # Conventional toolkit root inferred from the nvcc executable.
        cuda_home = Path(nvcc).resolve().parent.parent

        # Conda stores CUDA headers and libraries under this target-specific root.
        conda_cuda_home = cuda_home / "targets" / "x86_64-linux"

        # Directory containing Conda's CUDA runtime and driver-stub libraries.
        conda_cuda_lib = conda_cuda_home / "lib"
        if not (cuda_home / "lib64").exists() and conda_cuda_lib.exists():
            # Project-local toolkit facade with FlashInfer's expected directory names.
            cuda_compat_home = Path(__file__).resolve().parent / ".cuda_compat_conda"
            cuda_compat_home.mkdir(exist_ok=True)

            # Facade entries mapped to the actual Conda toolkit directories.
            cuda_compat_links = {
                "bin": Path(nvcc).resolve().parent,
                "lib64": conda_cuda_lib,
            }
            for link_name, link_target in cuda_compat_links.items():
                # Destination link under the compatibility toolkit root.
                link_path = cuda_compat_home / link_name
                if not link_path.exists():
                    link_path.symlink_to(link_target, target_is_directory=True)

            # Real include facade that can combine the CUDA toolkit and the
            # separately packaged cuRAND headers without mixing CUDA versions.
            cuda_compat_include = cuda_compat_home / "include"
            cuda_compat_include.mkdir(exist_ok=True)

            # Every header entry supplied by the Conda CUDA toolkit.
            conda_cuda_include = conda_cuda_home / "include"
            for header_source in conda_cuda_include.iterdir():
                # Header link presented beneath the compatibility include path.
                header_link = cuda_compat_include / header_source.name
                if not header_link.exists():
                    header_link.symlink_to(
                        header_source, target_is_directory=header_source.is_dir()
                    )
            cuda_home = cuda_compat_home

            # Library search paths needed because this Conda toolkit uses
            # ``lib`` while FlashInfer normally adds only ``lib64``.
            cuda_library_paths = [
                str(conda_cuda_lib),
                str(conda_cuda_lib / "stubs"),
            ]

            # Existing compiler library path supplied by the caller, if any.
            existing_library_path = os.environ.get("LIBRARY_PATH")
            if existing_library_path:
                cuda_library_paths.append(existing_library_path)
            os.environ["LIBRARY_PATH"] = os.pathsep.join(cuda_library_paths)

            # CUDA directories searched by the runtime dynamic loader.
            cuda_runtime_paths = [
                str(conda_cuda_lib),
                str(conda_cuda_lib / "stubs"),
            ]

            # Existing dynamic-loader path supplied by the caller, if any.
            existing_ld_library_path = os.environ.get("LD_LIBRARY_PATH")
            if existing_ld_library_path:
                cuda_runtime_paths.append(existing_ld_library_path)
            os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(cuda_runtime_paths)

            # Matching optional CUDA library headers supplied by Python packages.
            python_cuda_include_candidates = list(
                (Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}"
                 / "site-packages" / "nvidia").glob("cu*/include")
            )
            for python_cuda_include in python_cuda_include_candidates:
                if (python_cuda_include / "curand.h").exists():
                    for curand_source in python_cuda_include.glob("curand*.h"):
                        # cuRAND header link added without replacing core CUDA headers.
                        curand_link = cuda_compat_include / curand_source.name
                        if not curand_link.exists():
                            curand_link.symlink_to(curand_source)
                    break

        os.environ.setdefault("CUDA_HOME", str(cuda_home))
        os.environ.setdefault("FLASHINFER_NVCC", str(Path(nvcc).resolve()))

    # Parsed settings supplied by the user or by an internal worker command.
    args = parse_args()
    if args._worker_model:
        if not args._worker_label or args._raw_csv is None:
            raise SystemExit("Internal worker arguments are incomplete")
        run_worker(args._worker_model, args._worker_label, args._raw_csv, args.repeats)
        return

    # Location where this run's CSV, metadata, and figures will be stored.
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Shared raw-measurement file populated by all sequential workers.
    csv_path = args.output_dir / "raw_measurements.csv"
    if not args.plot_only:
        if args.repeats < 1:
            raise SystemExit("--repeats must be at least 1")
        # Model tuples selected by the optional --models filter.
        selected = select_models(args.models)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=CSV_FIELDS).writeheader()
        write_metadata(args.output_dir, selected, args.repeats)

        # Absolute path used to relaunch this same file in worker mode.
        script = Path(__file__).resolve()
        for model, label in selected:
            # Clean child-process invocation for exactly one checkpoint.
            command = [
                sys.executable,
                str(script),
                "--_worker-model",
                model,
                "--_worker-label",
                label,
                "--_raw-csv",
                str(csv_path.resolve()),
                "--repeats",
                str(args.repeats),
            ]
            # Completed worker process used to detect a failed checkpoint cleanly.
            completed = subprocess.run(command, check=False)
            if completed.returncode != 0:
                raise SystemExit(
                    f"Benchmark worker for {label} failed with exit code "
                    f"{completed.returncode}. Completed measurements remain in {csv_path}."
                )

    if not csv_path.exists():
        raise SystemExit(f"Cannot plot: {csv_path} does not exist")
    plot_results(csv_path, args.output_dir)
    print(f"\nRaw measurements: {csv_path}")
    print(f"Prefill figure:    {args.output_dir / 'prefill_throughput.png'}")
    print(f"Decode figure:     {args.output_dir / 'decode_throughput.png'}")


if __name__ == "__main__":
    main()
