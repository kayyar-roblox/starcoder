import os
import torch
from absl import app, flags
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "base_model_name_or_path", "bigcode/large-model", "Base model name or path."
)
flags.DEFINE_string("peft_model_path", "/", "PEFT model path.")


def main(argv):
    base_model = AutoModelForCausalLM.from_pretrained(
        FLAGS.base_model_name_or_path,
        return_dict=True,
        torch_dtype=torch.float16,
    )

    model = PeftModel.from_pretrained(base_model, FLAGS.peft_model_path)
    model = model.merge_and_unload()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.base_model_name_or_path)

    model.save_pretrained(f"{FLAGS.base_model_name_or_path}-merged")
    tokenizer.save_pretrained(f"{FLAGS.base_model_name_or_path}-merged")
    print(f"Model saved to {FLAGS.base_model_name_or_path}-merged")


if __name__ == "__main__":
    app.run(main)
