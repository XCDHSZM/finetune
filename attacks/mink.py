import numpy as np
import torch
import torch.nn.functional as F
from attacks import AbstractAttack
from datasets import Dataset
from transformers import PreTrainedModel


def min_k_prob(model: PreTrainedModel, token_ids: torch.Tensor, attention_mask: torch.Tensor, k: int = 20):
    with torch.no_grad():
        outputs = model(token_ids, attention_mask=attention_mask)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        
        # Get the vocab size
        if hasattr(model, 'module'):  # For DataParallel
            vocab_size = model.module.config.vocab_size
        else:
            vocab_size = model.config.vocab_size
        
        shift_logits = logits[..., :-1, :].contiguous().view(-1, vocab_size)
        shift_labels = token_ids[..., 1:].contiguous().view(-1)
        
        # Free memory
        del logits, outputs
        torch.cuda.empty_cache()
        
        token_probs = F.softmax(shift_logits, dim=-1)
        token_probs = torch.gather(token_probs, 1, shift_labels.unsqueeze(1)).squeeze(1)
        
        # Free memory
        del shift_logits
        torch.cuda.empty_cache()
        
        token_probs = token_probs.view(token_ids.shape[0], -1)
        token_probs = token_probs * attention_mask[:, 1:]
        
        sorted_probs, _ = torch.sort(token_probs, dim=-1)
        k_min_probs = sorted_probs[:, :k].sum(dim=-1) / attention_mask[:, 1:].sum(dim=-1)
        
        return k_min_probs.cpu().numpy()


class MinkAttack(AbstractAttack):
    def __init__(self, name, model, tokenizer, config, device):
        super().__init__(name, model, tokenizer, config, device)

    def run(self, dataset: Dataset) -> Dataset:
        dataset = dataset.map(
            lambda x: self.score(x),
            batched=True,
            batch_size=self.config.get('batch_size', 4),  # Reduced batch size
            new_fingerprint=f"{self.signature(dataset)}_v3",
        )
        return dataset

    def score(self, batch):
        texts = [x for x in batch['text']]
        tokenized = self.tokenizer.batch_encode_plus(
            texts, 
            return_tensors='pt', 
            padding=True,
            truncation=True,
            max_length=1024  # Add max length
        )
        token_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)
        k_min_probas = min_k_prob(self.model, token_ids, attention_mask, k=self.config.get('k', 20))
        
        # Clear CUDA cache
        del token_ids, attention_mask
        torch.cuda.empty_cache()
        
        return {self.name: k_min_probas}

    def extract_features(self, batch):
        # Get top-k losses
        k = self.config['k']
        texts = batch["text"]
        features = []
        
        for text in texts:
            # Get token-level losses
            tokens = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
            with torch.no_grad():
                outputs = self.model(tokens)
                losses = torch.nn.functional.cross_entropy(
                    outputs.logits[:, :-1, :].flatten(0, 1),
                    tokens[:, 1:].flatten(),
                    reduction='none'
                ).reshape(tokens.shape[0], -1)
            
            # Get top-k losses
            top_k_losses = torch.topk(losses[0], min(k, losses.shape[1])).values
            # Pad if necessary
            if len(top_k_losses) < k:
                top_k_losses = torch.cat([
                    top_k_losses,
                    torch.zeros(k - len(top_k_losses), device=self.device)
                ])
            features.append(top_k_losses.cpu().numpy())
            
        return np.array(features)
