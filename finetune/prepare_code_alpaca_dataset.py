import datasets
import cached_constant_length_dataset
from pathlib import Path
from transformers import AutoTokenizer
import argparse


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default="bigcode/starcoder")
    parser.add_argument(
        "--dataset", type=str, default="HuggingFaceH4/CodeAlpaca_20K"
    )
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    prepare(tokenizer, args.dataset, args.max_length)


if __name__ == "__main__":
    main()
