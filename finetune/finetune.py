"""
Fine-tunes a model on a given dataset.

Usage:
    python finetune.py --dataset_name HuggingFaceH4/CodeAlpaca_20K

The above command will fine-tune a model on the 'HuggingFaceH4/CodeAlpaca_20K' dataset.
"""


import os
from absl import app, flags
from transformers import AutoTokenizer, GPT2TokenizerFast, logging, set_seed
from dataset_types import ConstantLengthDataset, TestTrainDataset
from dataset_utils import create_datasets
from training_utils import run_training
import pdb

# Model and Dataset


def get_gpu_memory_in_mb():
    """Returns the currently available GPU memory in MB."""
    import torch

    # Return 0 if no GPU is available.
    if not torch.cuda.is_available():
        return 0

    ret = torch.cuda.get_device_properties(0).total_memory
    ret = ret / 1024 / 1024 // 1024
    return ret


def get_default_model():
    # Use bigcode/santacoder if GPU memory is below 32 Mb, bigcode/starcoder otherwise.
    if get_gpu_memory_in_mb() < 32 * 1024:
        return "bigcode/santacoder"

    return "bigcode/starcoder"


flags.DEFINE_string("model_path", get_default_model(), "Model path.")
flags.DEFINE_string(
    "dataset_name", "HuggingFaceH4/CodeAlpaca_20K", "Dataset name."
)
flags.DEFINE_string("subset", None, "Subset.")
flags.DEFINE_string("split", None, "Split.")

# Data Processing
flags.DEFINE_integer("size_valid_set", 10000, "Size of the validation set.")
flags.DEFINE_bool("streaming", False, "Streaming.")
flags.DEFINE_integer("shuffle_buffer", 5000, "Shuffle buffer size.")
flags.DEFINE_string("input_column_name", "prompt", "Input column name.")
flags.DEFINE_string("output_column_name", "completion", "Output column name.")

# Training
flags.DEFINE_integer("seq_length", 2048, "Sequence length.")
flags.DEFINE_integer("max_steps", 10000, "Max steps.")
flags.DEFINE_integer("batch_size", 1, "Batch size.")
flags.DEFINE_integer(
    "gradient_accumulation_steps", 16, "Gradient accumulation steps."
)
flags.DEFINE_integer("eos_token_id", 49152, "EOS token ID.")
flags.DEFINE_integer("lora_r", 16, "LoRA r.")
flags.DEFINE_integer("lora_alpha", 32, "LoRA alpha.")
flags.DEFINE_float("lora_dropout", 0.05, "LoRA dropout.")
flags.DEFINE_float("learning_rate", 5e-6, "Learning rate.")
flags.DEFINE_string("lr_scheduler_type", "cosine", "LR scheduler type.")
flags.DEFINE_integer("num_warmup_steps", 100, "Number of warmup steps.")
flags.DEFINE_float("weight_decay", 0.05, "Weight decay.")
flags.DEFINE_integer("local_rank", 0, "Local rank.")
flags.DEFINE_bool("no_fp16", True, "Disable FP16.")
flags.DEFINE_bool("bf16", True, "BF16.")
flags.DEFINE_bool(
    "no_gradient_checkpointing", True, "Disable gradient checkpointing."
)
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("num_workers", None, "Number of workers.")
flags.DEFINE_string("output_dir", "./checkpoints", "Output directory.")
flags.DEFINE_integer("log_freq", 100, "Log frequency.")
flags.DEFINE_integer("eval_freq", 100, "Evaluation frequency.")
flags.DEFINE_integer("save_freq", 1000, "Save frequency.")


def main(unused_args) -> None:
    del unused_args  # unused
    FLAGS = flags.FLAGS
    set_seed(FLAGS.seed)
    os.makedirs(FLAGS.output_dir, exist_ok=True)

    logging.set_verbosity_error()

    tokenizer: GPT2TokenizerFast = AutoTokenizer.from_pretrained(
        FLAGS.model_path, trust_remote_code=True, use_auth_token=True
    )
    train_dataset, eval_dataset = create_datasets(tokenizer, flags.FLAGS)
    run_training(flags.FLAGS, train_dataset, eval_dataset)


if __name__ == "__main__":
    app.run(main)
