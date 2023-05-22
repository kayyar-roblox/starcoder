from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)
from accelerate import Accelerator
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from utils import print_trainable_parameters
from absl import flags
from dataset_types import ConstantLengthDataset
import os
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import torch

class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        training_args: TrainingArguments,
        trainer_state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            training_args.output_dir,
            f"{PREFIX_CHECKPOINT_DIR}-{trainer_state.global_step}",
        )

        kwargs["model"].save_pretrained(checkpoint_folder)

        pytorch_model_path = os.path.join(
            checkpoint_folder, "pytorch_model.bin"
        )
        torch.save({}, pytorch_model_path)
        return control


class LoadBestPeftModelCallback(TrainerCallback):
    def on_train_end(
        self,
        training_args: TrainingArguments,
        trainer_state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        print(
            f"Loading best peft model from {trainer_state.best_model_checkpoint} (score: {trainer_state.best_metric})."
        )
        best_model_path = os.path.join(
            trainer_state.best_model_checkpoint, "adapter_model.bin"
        )
        adapters_weights = torch.load(best_model_path)
        model = kwargs["model"]
        set_peft_model_state_dict(model, adapters_weights)
        return control


def run_training(
    args: flags.FlagValues,
    train_data: ConstantLengthDataset,
    val_data: ConstantLengthDataset,
) -> None:
    print("Loading the model")
    # disable caching mechanism when using gradient checkpointing
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        use_auth_token=True,
        trust_remote_code=True,
        use_cache=not args.no_gradient_checkpointing,
        load_in_8bit=True,
        device_map={"": Accelerator().process_index},
    )
    model = prepare_model_for_int8_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_proj", "c_attn", "q_attn"],
    )

    model = get_peft_model(model, lora_config)

    print_trainable_parameters(model)

    train_data.start_iteration = 0

    print("Starting main loop")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        evaluation_strategy="steps",
        max_steps=args.max_steps,
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        logging_steps=args.log_freq,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        fp16=not args.no_fp16,
        bf16=args.bf16,
        weight_decay=args.weight_decay,
        run_name="StarCoder-finetuned",
        report_to="wandb",
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        callbacks=[SavePeftModelCallback, LoadBestPeftModelCallback],
    )

    print("Training...")
    trainer.train()

    print("Saving last checkpoint of the model")
    model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))
