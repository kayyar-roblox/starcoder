from tqdm import tqdm
from typing import Any, Dict
import os


def chars_token_ratio(
    dataset: Any,
    tokenizer: Any,
    input_column_name: str = "prompt",
    output_column_name: str = "completion",
    nb_examples: int = 400,
) -> float:
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(
        zip(range(nb_examples), iter(dataset)), total=nb_examples
    ):
        text = prepare_sample_text(
            example, input_column_name, output_column_name
        )
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def print_trainable_parameters(model: Any) -> None:
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_sample_text(
    example: Dict[str, Any],
    input_column_name: str = "prompt",
    output_column_name: str = "completion",
) -> str:
    """Prepare the text from a sample of the dataset."""
    text = f"Question: {example[input_column_name]}\n\nAnswer: {example[output_column_name]}"
    return text


def is_macos() -> bool:
    return os.uname().sysname == "Darwin"
