# Unit test for dataset_types.py

from dataset_types import ConstantLengthDataset
from absl.testing import absltest
from absl.testing import parameterized

import transformers


class ConstantLengthDatasetTest(absltest.TestCase):
    def setUp(self):
        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")

    def test_constant_length_dataset(self):
        dataset = ConstantLengthDataset(
            self.tokenizer,
            ["abc", "def", "ghi"],
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
        # TODO: Add a functional test.


if __name__ == "__main__":
    absltest.main()
