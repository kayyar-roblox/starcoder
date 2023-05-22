"""
Use this module to preprocess a fine-tuning dataset for a StarCoder model.
Preprocessing a dataset involves:

1. Formatting each item in the dataset for input;
2. concatenating short documents into a single input (separated by <|endoftext|>);
3. Splitting long documents into multiple items.

Doing this in a preprocessing step instead of dynamically makes it possible to
determine the number of epochs taken.

For an example of how to use this model to preprocess, see the file
prepare_code_alpaca_dataset.py.

The two main functions in this module are:
- save_prepared_dataset: preprocesses a dataset and saves it to a .jsonl file
- load_prepared_dataset: load a preprocessed dataset from a .jsonl file and
  and produces a dataset of items. This dataset can be used as:
   
  + The train_dataset argument for a Hugging Face Trainer class:
    
    https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer.train_dataset

  + or the eval_dataset argument for a Hugging Face Trainer class:

    https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer.eval_dataset
"""
from transformers import AutoTokenizer
from pathlib import Path
import json
from tqdm import tqdm
import datasets
import torch


def save_prepared_dataset(
    tokenizer: AutoTokenizer,
    src_dataset_generator,
    dst_path: Path,
    max_length: int,
    input_select_fn,
):
    """
    Preprocesses a dataset for fine-tuning. The arguments are:

    - tokenizer: the tokenizer for the model
    - src_dataset_generator: any generator that produces training items
    - dst_path: output path for the prepared dataset
    - max_length: maximum tokens per item, as determined by the model
    - input_select_fn: a function applied to each training item to extract
    - text
    """
    with dst_path.open("w") as output_file:
        dest_row = []
        dest_row_len = 0
        for chunk in chunk_dataset(
            tokenizer, src_dataset_generator, input_select_fn, max_length
        ):
            if dest_row_len + chunk["num_tokens"] > max_length:
                output_dest_row(output_file, max_length, dest_row)
                dest_row.clear()
                dest_row_len = 0
            dest_row.append(chunk)
            dest_row_len += chunk["num_tokens"]
        # Final row if needed.
        if len(dest_row) > 0:
            output_dest_row(output_file, max_length, dest_row)


def load_prepared_dataset(tokenizer, path: str):
    """
    Turns a prepared dataset into a dataset of items for training. Each item is:

    { "input_ids": torch.LongTensor, "labels": torch.LongTensor }
    """
    chunked_dataset = datasets.load_dataset(
        "json", data_files=path, split="train"
    )
    return chunked_dataset.map(
        lambda item: tokenize_cached_item(tokenizer, item),
        remove_columns=["chunks"],
    )


def chunk_dataset(
    tokenizer: AutoTokenizer, dataset, input_select_fn, max_length: int
):
    """
    A generator function that produces the text in every row of dataset,
    broken into chunks of length at most max_length (including the EOS token).
    Each item produced by the generator is a dictionary:

    { "eos": bool, "num_tokens": int, "text": str }

    If "eos" is True, then the end-of-text token should be appended to "text".
    """
    for source_row in tqdm(dataset, "Chunking dataset"):
        source_text = input_select_fn(source_row)
        # There are some uncommon options in the call to tokenizer that I'm
        # documenting below.
        #
        # When return_overflowing_tokens=True, the tokenizer produces a
        # list of list of tokens. This list will have length > 1 if the
        # length of the text exceeds max_length.
        #
        # When return_offsets_mapping=True, the tokenizer produces a list
        # of tuples that correspond to each token and indicate the span
        # of each token. When return_overflowing_tokens=True, this is a
        # list of list of tuples instead.
        tokenized_items = tokenizer(
            source_text,
            truncation=True,
            max_length=max_length,
            return_overflowing_tokens=True,
            return_attention_mask=False,
            return_offsets_mapping=True,
        )
        max_chunk_index = len(tokenized_items["offset_mapping"]) - 1
        for chunk_index, offsets in enumerate(
            tokenized_items["offset_mapping"]
        ):
            first_token_start = offsets[0][0]
            last_token_end = offsets[-1][1]
            is_final_chunk = chunk_index == max_chunk_index
            num_tokens = len(offsets) + (1 if is_final_chunk else 0)
            if is_final_chunk and num_tokens == max_length + 1:
                # Corner case: the chunk is exactly max_length tokens long, but we need to add
                # <|endoftext|>. As a *HACK* we remove the last token.
                last_token_end = offsets[-2][1]
                num_tokens -= 1
            assert num_tokens <= max_length  # should never happen
            yield {
                "eos": is_final_chunk,
                "num_tokens": num_tokens,
                "text": source_text[first_token_start:last_token_end],
            }


def output_dest_row(output_file, max_length: int, row):
    """
    Writes a row of chunks to the output file. We also write the max_length to
    facilitate padding in load_prepared_dataset.
    """
    json.dump({"chunks": row, "max_length": max_length}, output_file)
    output_file.write("\n")


def tokenize_cached_item(tokenizer, item):
    """
    Reads a row from a preprocessed dataset, tokenizes it, and returns returns
    results that can be fed into a Hugging Face Trainer class.
    """
    tokens = []
    max_length = item["max_length"]
    for chunk in item["chunks"]:
        new_tokens = tokenizer(chunk["text"], return_attention_mask=False)
        tokens.extend(new_tokens["input_ids"])
        if chunk["eos"]:
            tokens.append(tokenizer.eos_token_id)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    tokens.extend([pad_token_id] * (max_length - len(tokens)))
    return {
        "input_ids": torch.LongTensor(tokens),
        "labels": torch.LongTensor(tokens),
    }
