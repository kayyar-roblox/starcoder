import datasets
import cached_constant_length_dataset
from pathlib import Path
from transformers import AutoTokenizer
from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string("tokenizer", "bigcode/starcoder", "Tokenizer model")
flags.DEFINE_string("dataset", "HuggingFaceH4/CodeAlpaca_20K", "Dataset to use")
flags.DEFINE_integer("max_length", 2048, "Maximum length for the tokenizer")


def code_alpaca_format(row):
    return f"Question: {row['prompt']}\n\nAnswer: {row['completion']}"


def prepare(tokenizer, dataset, max_length):
    dataset = datasets.load_dataset(dataset)
    cached_constant_length_dataset.save_prepared_dataset(
        tokenizer,
        dataset["train"],
        Path("train.jsonl"),
        max_length,
        code_alpaca_format,
    )
    cached_constant_length_dataset.save_prepared_dataset(
        tokenizer,
        dataset["test"],
        Path("valid.jsonl"),
        max_length,
        code_alpaca_format,
    )


def main(argv):
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.tokenizer)
    prepare(tokenizer, FLAGS.dataset, FLAGS.max_length)


if __name__ == "__main__":
    app.run(main)
