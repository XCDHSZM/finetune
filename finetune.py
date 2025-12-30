import os
import copy
import yaml
import torch
import argparse
from datetime import datetime
import torch.nn.functional as F

from transformers import TrainerCallback, AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from datasets import Dataset
import deepspeed
from accelerate import Accelerator
from accelerate.logging import get_logger

import warnings
warnings.filterwarnings("ignore")

logger = get_logger("finetune", "info")
os.environ["WANDB_MODE"] = "disabled"


class FinetuneConfig:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.model_name = config['model']['name']
        self.split_model = config['model'].get('split_model', True)

        self.dataset_name = config['dataset']['name']
        self.dataset_split = config['dataset']['split']
        self.cache_path = config['dataset']['cache_path']
        self.dataset_config_name = config['dataset'].get('config_name', None)

        training_config = config['training']
        self.epochs = int(training_config['epochs'])
        self.save_epochs = int(training_config['save_epochs'])
        self.batch_size = int(training_config['batch_size'])
        self.gradient_accumulation_steps = int(training_config['gradient_accumulation_steps'])
        self.learning_rate = float(training_config['learning_rate'])
        self.eval_steps = int(training_config['eval_steps'])
        self.log_steps = int(training_config['log_steps'])
        self.gradient_checkpointing = bool(training_config.get('gradient_checkpointing', False))
        self.lr_scheduler_type = training_config.get('lr_scheduler_type', 'cosine')
        self.warmup_steps = int(training_config.get('warmup_steps', 0))
        self.weight_decay = float(training_config.get('weight_decay', 0.01))
        self.deepspeed = training_config.get('deepspeed', None)
        self.fp16 = training_config.get('fp16', False)
        self.bf16 = training_config.get('bf16', True)


def create_checkpoint_dir(args):
    split_name = args.dataset_split.replace("/", "_")
    checkpoint_base = os.path.join("checkpoints", f"{args.model_name.split('/')[-1]}-{split_name}")
    checkpoint_dir = os.path.join(checkpoint_base, f"{args.select_ratio}-" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


class CustomCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.trainer.train_dataset = self.trainer.select_epoch_data()
        self.trainer.train_dataloader = self.trainer.get_train_dataloader()
        print(f"Updating training set with {len(self.trainer.train_dataset)} samples.")


# Custom SFTTrainer
class CustomSFTTrainer(SFTTrainer):
    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer, save_epochs=5, block_size=2048, 
                 model_name=None, dataset_name=None, dataset_split=None, select_ratio=0.5, **kwargs):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            max_seq_length=block_size,
            dataset_text_field="text",
            **kwargs
        )
        self.save_epochs = save_epochs
        self.total_epochs = int(self.args.num_train_epochs)
        self.model_name = model_name.split('/')[-1] if model_name else "model"
        self.dataset_name = dataset_name if dataset_name else "dataset"
        self.split_name = dataset_split.replace("/", "_") if dataset_split else ""
        # Get batch size directly from deepspeed config
        self.batch_size = self.args.deepspeed.get("train_batch_size", 128)  # default to 128 if not found

        self.raw_train_dataset = copy.deepcopy(train_dataset)
        self.count_change = 0
        self.select_ratio = select_ratio

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval=None, *args, **kwargs):
        save_epochs = [0, 1, 2, 3]
        if epoch is not None and epoch in save_epochs:
            checkpoint_name = f"{self.model_name}-{self.dataset_name}-{self.split_name}-batch{self.batch_size}-epoch{epoch+1}"
            checkpoint_folder = os.path.join(self.args.output_dir, checkpoint_name)
            logger.info(f"Saving checkpoint at epoch {epoch+1} to {checkpoint_folder}")
            self.save_model(checkpoint_folder)

    def select_epoch_data(self):
        if self.count_change > 0:
            select_ids = list(range(len(self.raw_train_dataset)))
        else:
            scores = {}
            for idx, data in enumerate(self.raw_train_dataset):
                score = self.eval_sample_loss(data)
                scores[idx] = score
            sorted_scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}
            select_len = int(len(sorted_scores) * self.select_ratio)
            select_ids = list(sorted_scores.keys())[:select_len]
            print(f"*** Replace {len(select_ids)} samples to paraphrased samples.")

        self.count_change += 1

        new_dataset = {"input_ids": [], "attention_mask": []}
        for idx in range(len(self.raw_train_dataset)):
            if idx in select_ids:
                input_ids = self.raw_train_dataset[idx]["para_input_ids"]
                attention_mask = self.raw_train_dataset[idx]["para_attention_mask"]
            else:
                input_ids = self.raw_train_dataset[idx]["input_ids"]
                attention_mask = self.raw_train_dataset[idx]["attention_mask"]

            new_dataset["input_ids"].append(input_ids)
            new_dataset["attention_mask"].append(attention_mask)

        return Dataset.from_dict(new_dataset)

    def eval_sample_loss(self, data):
        with torch.no_grad():
            token_ids, attention_mask = data["input_ids"], data["attention_mask"]
            # Convert list of tensors to single tensor
            token_ids = torch.tensor(token_ids).unsqueeze(0).to(self.model.device)
            attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(self.model.device)

            labels = token_ids.clone()
            outputs = self.model(token_ids, attention_mask=attention_mask)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            if isinstance(self.model, torch.nn.DataParallel):
                vocab_size = self.model.module.config.vocab_size
            else:
                vocab_size = self.model.config.vocab_size

            shift_logits = outputs.logits[..., :-1, :].contiguous().view(-1, vocab_size)
            shift_attention_mask = attention_mask[..., :-1]
            shift_targets = labels[..., 1:]
            shift_targets[shift_attention_mask == 0] = -100

            loss = F.cross_entropy(shift_logits, shift_targets.contiguous().view(-1), reduction="none")
            loss = loss.view(token_ids.shape[0], -1)
            loss = loss.sum(axis=1) / shift_attention_mask.sum(axis=1)

            return loss.detach().cpu().float().numpy()[0]


def validate_training_setup(model):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found in the model!")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    if trainable_params != total_params:
        raise ValueError("Not all parameters are trainable in fine-tuning!")


def get_optimizer_grouped_parameters(model, weight_decay, learning_rate):
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                        if (not any(nd in n for nd in no_decay))],
            "weight_decay": weight_decay,
            "lr": learning_rate
        },
        {
            "params": [p for n, p in model.named_parameters()
                        if (any(nd in n for nd in no_decay))],
            "weight_decay": 0.0,
            "lr": learning_rate
        },
    ]
    return optimizer_grouped_parameters


def get_optimizer(model, args):
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, args.weight_decay, args.learning_rate)

    if args.deepspeed:
        optimizer = None
    else:
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=args.weight_decay
        )
    return optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--select_ratio", type=float, default=0.5, help="Ratio of samples to select")
    args = parser.parse_args()

    config = FinetuneConfig(args.config)
    for key, value in vars(config).items():
        setattr(args, key, value)

    deepspeed.init_distributed()
    args.output_dir = create_checkpoint_dir(args)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        project_dir=args.output_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9)

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    from peft import get_peft_model, LoraConfig
    lora_config = LoraConfig(
        r=128,
        lora_alpha=256,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    for param in model.parameters():
        param.requires_grad = True

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    validate_training_setup(model)

    with accelerator.main_process_first():
        from data.prepare import dataset_prepare
        train_dataset, valid_dataset = dataset_prepare(args, tokenizer=tokenizer)
        train_dataset = Dataset.from_dict(train_dataset[0:])
        valid_dataset = Dataset.from_dict(valid_dataset[0:])

    optimizer = get_optimizer(model, args)

    total_train_batch_size = (
        args.batch_size
        * args.gradient_accumulation_steps
        * torch.distributed.get_world_size()
    )
    num_update_steps_per_epoch = len(train_dataset) // total_train_batch_size
    max_train_steps = args.epochs * num_update_steps_per_epoch

    ds_config = args.deepspeed
    if ds_config is not None:
        ds_config['gradient_accumulation_steps'] = args.gradient_accumulation_steps
        ds_config['train_micro_batch_size_per_gpu'] = args.batch_size
        ds_config['train_batch_size'] = args.batch_size * args.gradient_accumulation_steps * torch.distributed.get_world_size()

        if 'optimizer' in ds_config:
            ds_config['optimizer']['params']['lr'] = args.learning_rate
            ds_config['optimizer']['params']['weight_decay'] = args.weight_decay

        if 'scheduler' in ds_config:
            ds_config['scheduler']['params']['warmup_num_steps'] = args.warmup_steps
            ds_config['scheduler']['params']['total_num_steps'] = max_train_steps

    deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config,
        model_parameters=get_optimizer_grouped_parameters(model, args.weight_decay, args.learning_rate)
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=False,
        dataloader_drop_last=True,
        eval_strategy="no",
        save_strategy="no",
        logging_strategy="steps",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=args.bf16,
        fp16=args.fp16,
        deepspeed=args.deepspeed,
        local_rank=args.local_rank,
        remove_unused_columns=False,
        label_names=["input_ids", "attention_mask"],
        ddp_find_unused_parameters=False,
        warmup_steps=args.warmup_steps,
        max_steps=max_train_steps,
        lr_scheduler_type=args.lr_scheduler_type
    )

    trainer = CustomSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        save_epochs=args.save_epochs,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        select_ratio=args.select_ratio,
    )

    trainer.add_callback(CustomCallback(trainer))
    trainer.train()
    trainer.save_model(f"{args.output_dir}/final")


if __name__ == "__main__":
    main()