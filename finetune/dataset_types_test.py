# Unit test for dataset_types.py

from dataset_types import ConstantLengthDataset
from absl.testing import absltest
from absl.testing import parameterized

import transformers
import pdb


class ConstantLengthDatasetTest(absltest.TestCase):
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
        # self.assertEqual(expected_iterations, actual_iterations)

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
