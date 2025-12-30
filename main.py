import argparse
import logging
import os
import time
import json
from collections import defaultdict
import numpy as np
import torch
from attacks.utils import batch_nlloss
from datasets import Dataset, load_dataset
from sklearn.metrics import auc, roc_curve
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import get_available_attacks, load_attack, load_config, load_mimir_dataset, set_seed

logging.basicConfig(level=logging.INFO)

def init_model(model_name, device):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except (OSError, EnvironmentError):
        base_model = "EleutherAI/pythia-6.9b"
        logging.warning(f"Could not load tokenizer from {model_name}. Falling back to base model {base_model}")
        tokenizer = AutoTokenizer.from_pretrained(base_model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16
    )

    model = model.to('cuda:0')

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    return model, tokenizer, device

def get_header(config):
    header = ["MIA", "AUC"]
    for t in config["fpr_thresholds"]:
        header.append(f"TPR@FPR={t}")
    return header

def get_printable_ds_name(ds_info):
    if "name" in ds_info:
        name = ds_info["name"]
    elif "mimir_name" in ds_info:
        name = ds_info["mimir_name"]
    else:
        raise ValueError()
    name = f"{name}/{ds_info['split']}"
    name = name.replace("/", "_")
    return name

def init_dataset(ds_info, model, tokenizer, device, batch_size, test_samples=None):
    if "mimir_name" in ds_info:
        if "name" in ds_info:
            raise ValueError("Cannot specify both 'name' and 'mimir_name' in dataset config")
        dataset = load_mimir_dataset(name=ds_info["mimir_name"], split=ds_info["split"])
    elif "name" in ds_info:
        dataset = load_dataset(ds_info["name"], split=ds_info["split"])
        dataset = dataset.shuffle(seed=42)
    else:
        raise ValueError("Dataset name is missing")

    if test_samples is not None and test_samples < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(test_samples))

    def process_batch(batch):
        nlloss_batch = batch_nlloss(batch, model, tokenizer, device)
        return {**batch, **nlloss_batch}

    dataset = dataset.map(
        process_batch,
        batched=True,
        batch_size=batch_size,
        remove_columns=[col for col in dataset.column_names if col not in ['label', 'text']],
        load_from_cache_file=False,
        num_proc=1
    )
    return dataset

def results_with_bootstrapping(y_true, y_pred, fpr_thresholds, n_bootstraps=1000):
    n = len(y_true)
    aucs = []
    tprs = {}
    for _ in range(n_bootstraps):
        idx = np.random.choice(n, n, replace=True)
        fpr, tpr, _ = roc_curve(np.array(y_true)[idx], np.array(y_pred)[idx])
        aucs.append(auc(fpr, tpr))
        for t in fpr_thresholds:
            if t not in tprs.keys():
                tprs[t] = [tpr[np.argmin(np.abs(fpr - t))]]
            else:
                tprs[t].append(tpr[np.argmin(np.abs(fpr - t))])

    results = [f"{np.mean(aucs): .3f} ± {np.std(aucs):.3f}"] + \
        [f"{np.mean(tprs[t]): .3f} ± {np.std(tprs[t]):.3f}" for t in fpr_thresholds]
    return results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('--run-all', action='store_true')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--target-model', type=str, required=True)
    parser.add_argument('--seed', type=int, help='Random seed', default=None)
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., arxiv)')
    parser.add_argument('--split', type=str, required=True, help='Split name (e.g., ngram_13_0.8)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S")

    if args.seed is not None:
        set_seed(args.seed)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    config = load_config(args.config)
    global_config = config['global']
    device = torch.device(global_config["device"])

    model_path = args.target_model if args.target_model else global_config['target_model']
    print(f"Loading model from {model_path}")
    model, tokenizer, device = init_model(model_path, device)

    results_to_print = {}
    config['global']['datasets'][0]['mimir_name'] = args.dataset
    config['global']['datasets'][0]['split'] = args.split

    for ds_info in global_config['datasets']:
        dataset = init_dataset(
            ds_info=ds_info,
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=global_config["batch_size"],
            test_samples=config["global"].get("test_samples")
        )
        ds_name = get_printable_ds_name(ds_info)
        
        results = []
        header = get_header(global_config)

        attack_names = ""
        results_file = f"{args.output}/{config['global'].get('test_samples')}samples_{current_time}_{ds_name}_attack_results.txt"
        with open(results_file, "w") as f:
            f.write(tabulate([], headers=header, tablefmt="outline") + "\n")

        for attack_name in get_available_attacks(config):
            logging.info(f"Running {attack_name} on {ds_name}")

            attack = load_attack(attack_name, model, tokenizer, config[attack_name], device)
            dataset = attack.run(dataset)

            if 'label' not in dataset.column_names:
                raise ValueError(f"Dataset does not contain 'label' column after running {attack_name} attack")

            y_true = dataset['label']
            y_score = dataset[attack_name]
            attack_results = results_with_bootstrapping(
                y_true,
                y_score,
                fpr_thresholds=global_config["fpr_thresholds"],
                n_bootstraps=global_config["n_bootstrap_samples"]
            )

            attack_row = [attack_name] + attack_results
            with open(results_file, "a") as f:
                table = tabulate([attack_row], tablefmt="outline")
                table_without_header = '\n'.join(table.split('\n')[1:])
                f.write(table_without_header)

            results.append(attack_row)
            logging.info(f"AUC {attack_name} on {ds_name}: {attack_results[0]}")
            attack_names += f"{attack_name}_"

        results_to_print[ds_name] = tabulate(results, headers=header, tablefmt="outline")

    with open(f"{args.output}/{config['global'].get('test_samples')}samples_{current_time}_{ds_name}_{attack_names}results.txt", "w") as f:
        f.write(tabulate(results, headers=header, tablefmt="outline"))

    for ds_name, res in results_to_print.items():
        print(f"Dataset: {ds_name}")
        print(res)