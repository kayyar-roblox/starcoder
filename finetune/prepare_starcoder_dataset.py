"""
This script prepares a subset of the StarCoder dataset for use with
cached_constant_length_dataset.py.

Space requirements:

- The script will download the entire subset without streaming. Ensure that the
  HF_DATASETS_CACHE environment variable is set to a directory with sufficient
  space. (The default is ~/.cache.)
- We create train.jsonl and valid.jsonl that will be 3-5x larger than the subset,
  which is compressed. So, ensure that this directory is on a volume with
  sufficient space.
"""
import datasets
import cached_constant_length_dataset
from pathlib import Path
from transformers import AutoTokenizer
from absl import app
from absl import flags
from typing import TypedDict, List

FLAGS = flags.FLAGS

flags.DEFINE_string("tokenizer", "bigcode/starcoder", "Tokenizer model")
flags.DEFINE_string("dataset", "bigcode/starcoderdata", "Dataset to use")
flags.DEFINE_string("data_dir", "lua", "datadir argument")
flags.DEFINE_integer("max_length", 2048, "Maximum length for the tokenizer")
flags.DEFINE_float(
    "validation_size", 0.01, "Percentage of dataset to use for validation"
)


class StarCoderDataItem(TypedDict):
    content: str


def starcoderdata_format(item: StarCoderDataItem):
    return item["content"]


def prepare(
    tokenizer: AutoTokenizer,
    dataset: str,
    data_dir: str,
    max_length: int,
    validation_size: float,
):
    # There is just a train split. IIRC, the validation set was dynamically selected at the
    # start of every run.
    dataset = datasets.load_dataset(dataset, data_dir=data_dir, split="train")
    dataset = dataset.train_test_split(test_size=validation_size, shuffle=True)

    cached_constant_length_dataset.save_prepared_dataset(
        tokenizer,
        dataset["test"],
        Path("valid.jsonl"),
        max_length,
        starcoderdata_format,
    )
    cached_constant_length_dataset.save_prepared_dataset(
        tokenizer,
        dataset["train"],
        Path("train.jsonl"),
        max_length,
        starcoderdata_format,
    )


def main(argv: List[str]):
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.tokenizer)
    prepare(
        tokenizer,
        FLAGS.dataset,
        FLAGS.data_dir,
        FLAGS.max_length,
        FLAGS.validation_size,
    )


if __name__ == "__main__":
    app.run(main)
