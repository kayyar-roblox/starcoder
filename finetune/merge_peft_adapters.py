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

    # Make sure GPU is being used when available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(FLAGS.base_model_name_or_path)
    merged_model_path = f"{FLAGS.base_model_name_or_path}-{FLAGS.peft_model_path}"

    model.save_pretrained(merged_model_path)
    tokenizer.save_pretrained(merged_model_path)
    print(f"Model saved to {merged_model_path}")


if __name__ == "__main__":
    app.run(main)
