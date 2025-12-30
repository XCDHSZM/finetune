from typing import Optional

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
import random


def compute_nlloss(
    model: PreTrainedModel,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    ignore_prefix: Optional[int] = None,
):
    with torch.no_grad():
        labels = token_ids.clone()

        outputs = model(token_ids, attention_mask=attention_mask)
        
        # Handle DataParallel output
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        if isinstance(model, torch.nn.DataParallel):
            vocab_size = model.module.config.vocab_size
        else:
            vocab_size = model.config.vocab_size

        shift_logits = outputs.logits[..., :-1, :].contiguous().view(-1, vocab_size)
        shift_attention_mask = attention_mask[..., :-1]
        shift_targets = labels[..., 1:]

        shift_targets[shift_attention_mask == 0] = -100

        loss = F.cross_entropy(shift_logits, shift_targets.contiguous().view(-1), reduction="none")
        loss = loss.view(token_ids.shape[0], -1)

        if ignore_prefix:
            loss = loss[:, ignore_prefix:]
            shift_attention_mask = shift_attention_mask[:, ignore_prefix:]

        loss = loss.sum(axis=1) / shift_attention_mask.sum(axis=1)

        return loss.detach().cpu().numpy()


def batch_nlloss(batch, model, tokenizer, device, key='nlloss'):
    texts = batch['text']
    tokenized = tokenizer.batch_encode_plus(texts, return_tensors='pt', padding=True, truncation=True)
    token_ids = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)
    
    # Split the batch into smaller chunks
    chunk_size = 1  # Process one sample at a time
    losses = []
    for i in range(0, len(texts), chunk_size):
        chunk_ids = token_ids[i:i+chunk_size]
        chunk_mask = attention_mask[i:i+chunk_size]
        chunk_losses = compute_nlloss(model, chunk_ids, chunk_mask)
        losses.extend(chunk_losses)
    
    return {key: losses}


def make_recall_prefix(dataset, n_shots, perplexity_bucket=None):
    """Create a prefix from random samples in the dataset."""
    if perplexity_bucket is not None:
        dataset = dataset.filter(lambda x: x["perplexity_bucket"] == perplexity_bucket)
    
    indices = random.sample(range(len(dataset)), n_shots)
    prefixes = [dataset[i]["text"] for i in indices]
    
    return " ".join(prefixes)
