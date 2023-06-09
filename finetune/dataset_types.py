from torch.utils.data import IterableDataset
from typing import Iterable, Tuple
from transformers import GPT2TokenizerFast
from utils import prepare_sample_text
import torch
import time
import pdb

TestTrainDataset = Tuple[IterableDataset, IterableDataset]


class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
    """

    def __init__(
        self,
        tokenizer: GPT2TokenizerFast,
        dataset: Iterable,
        infinite: bool = False,
        seq_length: int = 1024,
        num_of_sequences: int = 1024,
        chars_per_token: float = 3.6,
        input_column_name: str = "prompt",
        output_column_name: str = "completion",
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = (
            tokenizer.eos_token_id
            if tokenizer.eos_token_id is not None
            else args.eos_token_id
        )
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.input_column_name = input_column_name
        self.output_column_name = output_column_name
        self.epoch = 0

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    sample_text = prepare_sample_text(
                        next(iterator),
                        self.input_column_name,
                        self.output_column_name,
                    )
                    buffer.append(sample_text)
                    buffer_len += len(sample_text)
                except StopIteration:
                    if self.infinite:
                        self.epoch += 1
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(buffer, truncation=False)[
                "input_ids"
            ]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                # Pad the last element if necessary
                if len(input_ids) < self.seq_length:
                    num_padding_tokens = self.seq_length - len(input_ids)
                    input_ids.extend(
                        [self.concat_token_id] * num_padding_tokens
                    )
                assert len(input_ids) == self.seq_length
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(input_ids),
                    "labels": torch.LongTensor(input_ids),
                }
