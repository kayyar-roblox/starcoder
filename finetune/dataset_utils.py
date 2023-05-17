from datasets import load_dataset
from dataset_types import ConstantLengthDataset, TestTrainDataset
from transformers import (
    GPT2TokenizerFast,
)
from absl import flags
from utils import chars_token_ratio


def create_datasets(
    tokenizer: GPT2TokenizerFast, args: flags.FlagValues
) -> TestTrainDataset:
    dataset = load_dataset(
        args.dataset_name,
        data_dir=args.subset,
        split=args.split,
        use_auth_token=True,
        num_proc=args.num_workers if not args.streaming else None,
        streaming=args.streaming,
    )
    if args.streaming:
        print("Loading the dataset in streaming mode")
        valid_data = dataset.take(args.size_valid_set)
        train_data = dataset.skip(args.size_valid_set)
        train_data = train_data.shuffle(
            buffer_size=args.shuffle_buffer, seed=args.seed
        )
    else:
        train_data = dataset["train"]
        valid_data = dataset["test"]
        print(
            f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}"
        )

    chars_per_token = chars_token_ratio(
        train_data, tokenizer, args.input_column_name, args.output_column_name
    )
    print(
        f"The character to token ratio of the dataset is: {chars_per_token:.2f}"
    )

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
        input_column_name=args.input_column_name,
        output_column_name=args.output_column_name,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
        input_column_name=args.input_column_name,
        output_column_name=args.output_column_name,
    )
    return train_dataset, valid_dataset
