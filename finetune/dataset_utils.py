from datasets import load_dataset
from dataset_types import ConstantLengthDataset, TestTrainDataset
from transformers import (
    GPT2TokenizerFast,
)
from absl import flags
from utils import chars_token_ratio
import cached_constant_length_dataset

def create_datasets(
    tokenizer: GPT2TokenizerFast, args: flags.FlagValues
) -> TestTrainDataset:
    train_dataset = cached_constant_length_dataset.load_prepared_dataset(
        tokenizer, "train.jsonl"
    )
    valid_dataset = cached_constant_length_dataset.load_prepared_dataset(
        tokenizer, "valid.jsonl"
    )
    return train_dataset, valid_dataset
