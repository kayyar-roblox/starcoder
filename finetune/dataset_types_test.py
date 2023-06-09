# Unit test for dataset_types.py

from dataset_types import ConstantLengthDataset
from absl.testing import absltest
from absl.testing import parameterized
from torch.utils.data import IterableDataset
from utils import prepare_sample_text
import numpy as np
import transformers
import pdb


class MockDataset(IterableDataset):
    def __iter__(self):
        yield {
            "prompt": "What is your name?",
            "completion": "My name is ROBOT.",
        }
        yield {
            "prompt": "How old are you?",
            "completion": "I don't have an age.",
        }
        yield {
            "prompt": "What game is your favorite?",
            "completion": "Roblox, for sure :)",
        }


class ConstantLengthDatasetTest(parameterized.TestCase):
    def setUp(self):
        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")

    def testConstantLengthDataset_emitsTensors(self):
        dataset = ConstantLengthDataset(
            self.tokenizer,
            [
                {"prompt": "abc", "completion": "def"},
                {"prompt": "ghi", "completion": "jkl"},
                {"prompt": "mno", "completion": "pqr"},
            ],
            infinite=False,
            seq_length=1024,
            num_of_sequences=1024,
            chars_per_token=3.6,
            input_column_name="prompt",
            output_column_name="completion",
        )

        self.assertEqual(dataset.seq_length, 1024)
        self.assertEqual(dataset.infinite, False)
        self.assertEqual(dataset.current_size, 0)
        self.assertEqual(dataset.max_buffer_size, 1024 * 3.6 * 1024)
        self.assertEqual(dataset.input_column_name, "prompt")
        self.assertEqual(dataset.output_column_name, "completion")
        expected_iterations = 1
        actual_iterations = 0
        expected_keys = set(["input_ids", "labels"])

        for tensors in dataset:
            actual_iterations += 1
            self.assertEqual(tensors.keys(), expected_keys)
        self.assertEqual(expected_iterations, actual_iterations)

    @parameterized.parameters(np.arange(5, 30))
    def testConstantLengthDataset_checkNumSamplesAndCorrectness(
        self, seq_length
    ):
        # Create an instance of ConstantLengthDataset
        mock_dataset = MockDataset()
        constant_length_dataset = ConstantLengthDataset(
            self.tokenizer, dataset=mock_dataset, seq_length=seq_length
        )

        # 1. Make sure the actual number of generated examples match the expected number.
        expected_string = ""
        for sample in mock_dataset:
            expected_string += prepare_sample_text(sample)
            expected_string += self.tokenizer.eos_token
        tokenized_expected_string = self.tokenizer(expected_string)["input_ids"]
        tokens_in_last_sample = len(tokenized_expected_string) % seq_length
        num_padding_tokens = (
            0
            if tokens_in_last_sample == 0
            else seq_length - tokens_in_last_sample
        )
        expected_num_samples = len(tokenized_expected_string) // seq_length + (
            tokens_in_last_sample > 0
        )
        expected_string += self.tokenizer.eos_token * num_padding_tokens
        samples = list(constant_length_dataset)
        self.assertEqual(len(samples), expected_num_samples)

        # 2. Make sure the expected_string is the same as the actual_string.
        actual_string = ""
        for sample in samples:
            input_ids = sample["input_ids"]
            assert len(input_ids) == seq_length
            actual_string += self.tokenizer.decode(input_ids)
        self.assertEqual(actual_string, expected_string)

    def testInfiniteDataset_incrementsEpoch(self):
        dataset = ConstantLengthDataset(
            self.tokenizer,
            [
                {"prompt": "abc", "completion": "def"},
            ],
            infinite=True,
            seq_length=1024,
            num_of_sequences=1024,
            chars_per_token=3.6,
            input_column_name="prompt",
            output_column_name="completion",
        )

        self.assertEqual(dataset.infinite, True)
        self.assertEqual(dataset.epoch, 0)
        tensors = next(iter(dataset))
        expected_keys = set(["input_ids", "labels"])
        self.assertEqual(tensors.keys(), expected_keys)
        self.assertEqual(dataset.epoch, 145187)
        tensors = next(iter(dataset))
        self.assertEqual(dataset.epoch, 290374)


if __name__ == "__main__":
    absltest.main()
