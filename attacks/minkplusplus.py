import numpy as np
import torch
import torch.nn.functional as F
from attacks import AbstractAttack
from datasets import Dataset
from transformers import PreTrainedModel


def min_k_plusplus(model: PreTrainedModel, token_ids: torch.Tensor, attention_mask: torch.Tensor, k: int = 20):
    batch_size = token_ids.size(0)
    sub_batch_size = 8  # Adjust this based on your GPU memory
    k_min_logp_list = []

    for i in range(0, batch_size, sub_batch_size):
        end_idx = min(i + sub_batch_size, batch_size)
        batch_tokens = token_ids[i:end_idx]
        batch_attention = attention_mask[i:end_idx]

        with torch.no_grad():
            outputs = model(batch_tokens, attention_mask=batch_attention)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch_tokens[..., 1:].contiguous()
            
            # Get the vocab size
            vocab_size = model.module.config.vocab_size if hasattr(model, 'module') else model.config.vocab_size
            
            token_logp = -F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
                reduction='none'
            ).view(shift_logits.shape[:-1])
            
            token_logp = token_logp * batch_attention[:, 1:]
            
            sorted_logp, _ = torch.sort(token_logp, dim=-1)
            k_min_logp = sorted_logp[:, :k].sum(dim=-1) / batch_attention[:, 1:].sum(dim=-1)
            k_min_logp_list.append(k_min_logp.cpu())

            # Clean up GPU memory
            del outputs, logits, shift_logits, shift_labels, token_logp, sorted_logp
            torch.cuda.empty_cache()

    return torch.cat(k_min_logp_list, dim=0).numpy()


class MinkplusplusAttack(AbstractAttack):
    def __init__(self, name, model, tokenizer, config, device):
        super().__init__(name, model, tokenizer, config, device)
        self.vocab_size = model.module.config.vocab_size if hasattr(model, 'module') else model.config.vocab_size
        self.batch_size = config.get('batch_size', 32)
        
    def run(self, dataset: Dataset) -> Dataset:
        all_scores = []
        
        # Process dataset in chunks to avoid memory issues
        for i in range(0, len(dataset), self.batch_size):
            batch = dataset[i:i + self.batch_size]
            scores = self.score(batch)
            all_scores.extend(scores[self.name])
            
            # Clean up memory
            torch.cuda.empty_cache()
        
        return Dataset.from_dict({
            'text': dataset['text'],
            'label': dataset['label'],
            self.name: all_scores
        })

    def score(self, batch):
        texts = [x for x in batch['text']]
        tokenized = self.tokenizer.batch_encode_plus(
            texts, 
            return_tensors='pt', 
            padding="longest",
            truncation=True,
            max_length=2048  # Add explicit max length
        )
        
        token_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)
        
        k_min_probas = min_k_plusplus(
            self.model, 
            token_ids, 
            attention_mask, 
            k=self.config['k']
        )
        
        # Clean up GPU memory
        del token_ids, attention_mask, tokenized
        torch.cuda.empty_cache()
        
        return {self.name: k_min_probas}

    def extract_features(self, batch):
        # Get top-k and bottom-k losses
        k = self.config['k']
        texts = batch["text"]
        features = []
        
        for text in texts:
            # Get token-level losses
            tokens = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
            with torch.no_grad():
                outputs = self.model(tokens)
                # Calculate token-level losses
                losses = torch.nn.functional.cross_entropy(
                    outputs.logits[:, :-1, :].flatten(0, 1),
                    tokens[:, 1:].flatten(),
                    reduction='none'
                ).reshape(tokens.shape[0], -1)
            
            # Get top-k (highest) losses
            top_k_losses = torch.topk(losses[0], min(k, losses.shape[1])).values
            
            # Get bottom-k (lowest) losses
            bottom_k_losses = torch.topk(losses[0], min(k, losses.shape[1]), largest=False).values
            
            # Pad if necessary
            if len(top_k_losses) < k:
                top_k_losses = torch.cat([
                    top_k_losses,
                    torch.zeros(k - len(top_k_losses), device=self.device)
                ])
            if len(bottom_k_losses) < k:
                bottom_k_losses = torch.cat([
                    bottom_k_losses,
                    torch.zeros(k - len(bottom_k_losses), device=self.device)
                ])
            
            # Combine top-k and bottom-k losses
            combined_features = torch.cat([top_k_losses, bottom_k_losses])
            features.append(combined_features.cpu().numpy())
            
        return np.array(features)
