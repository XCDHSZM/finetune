import importlib
import random
from typing import Any, Dict
import numpy as np
import torch
import yaml
from datasets import Dataset, load_dataset

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as config_file:
        return yaml.safe_load(config_file)

def load_attack(name, model, tokenizer, config, device):
    module = importlib.import_module(f"attacks.{config['module']}")

    class_name = ''.join(word.capitalize() for word in name.split('_')) + 'Attack'
    class_name = class_name.replace('OfWords', 'ofWords')

    attr = getattr(module, class_name)
    return attr(
        name=name,
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=device
    )

def get_available_attacks(config):
    return [k for k in config.keys() if k != 'global']

def load_mimir_dataset(name: str, split: str) -> Dataset:
    dataset = load_dataset("iamgroot42/mimir", name, split=split)

    if 'label' not in dataset.column_names:
        if 'member' in dataset.column_names and 'nonmember' in dataset.column_names:
            all_texts = [dataset['member'][k] for k in range(len(dataset))]
            all_labels = [1] * len(dataset)
            all_texts += [dataset['nonmember'][k] for k in range(len(dataset))]
            all_labels += [0] * len(dataset)

            new_dataset = Dataset.from_dict({"text": all_texts, "label": all_labels})
            return new_dataset
        else:
            raise ValueError("Dataset does not contain 'label' column and cannot be inferred from 'member'/'nonmember' columns")
    
    return dataset

