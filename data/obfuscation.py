import os

from datasets import load_dataset, Dataset
from openai import OpenAI
from transformers import AutoTokenizer
from tqdm import tqdm


class OpenAIPrompter(object):

    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)

    def _pack(self, original_text: str):
        message = [
            {"role": "system", "content": "You are a helpful text rewriting assistant."},
            {"role": "user", "content":
             f"Rewrite the following paragraph by replacing every word with an alternative term that does not share the same root or spelling. Preserve the same meaning and sentence structure as much as possible.\n\"\"\"\n{original_text}\n\"\"\""},
        ]



        return message

    def query(self, original_text: str, **kwargs):
        message = self._pack(original_text)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=message,
            **kwargs
        )
        return response

    @staticmethod
    def parse_response(response):
        content = response.choices[0].message.content
        try:
            content = content.split('\"\"\"')[1]
        except IndexError:
            content = content
        return content


def load_raw_dataset(dataset_name, name, split_name, max_samples=None):
    dataset = load_dataset(dataset_name, name=name, split=split_name)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def build_editing_dataset(raw_dataset, prompt_ratio=0.5, max_length=1024):
    """
    Build editing dataset using neighbor-based paraphrasing.

    Returns dataset with:
    - prompt: First part of member text
    - original_response: Second part of member text
    - target_new: Paraphrased response from member neighbors
    - generality_prompt: Same as prompt
    - locality_prompt: First part of non-member text
    - locality_response: Second part of non-member text
    """
    results = {
        'prompt': [],
        'original_response': [],
        'target_new': [],
        'generality_prompt': [],
        'locality_prompt': [],
        'locality_response': []
    }

    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-6.9b')

    for member, nonmember, member_neighbors, nonmember_neighbors in zip(
        tqdm(raw_dataset['member'], desc='Processing Editing Dataset...'),
        raw_dataset['nonmember'],
        raw_dataset['member_neighbors'],
        raw_dataset['nonmember_neighbors']
    ):
        # prompt, original_response are split from member
        tokenized_member = tokenizer.encode(member, add_special_tokens=False, truncation=True, max_length=max_length)
        prompt_length = int(prompt_ratio * len(tokenized_member))
        prompt = tokenizer.decode(tokenized_member[:prompt_length])
        original_response = tokenizer.decode(tokenized_member[prompt_length:])

        # target_new is split from member_neighbors
        tokenized_member_neighbors = tokenizer.encode(member_neighbors[0], add_special_tokens=False, truncation=True, max_length=max_length)
        target_length = int(prompt_ratio * len(tokenized_member_neighbors))
        target_new = tokenizer.decode(tokenized_member_neighbors[target_length:])

        # generality_prompt is prompt (can be further mapped in later steps)
        generality_prompt = prompt

        # locality_prompt, locality_response are split from nonmember
        tokenized_nonmember = tokenizer.encode(nonmember, add_special_tokens=False, truncation=True, max_length=max_length)
        locality_length = int(prompt_ratio * len(tokenized_nonmember))
        locality_prompt = tokenizer.decode(tokenized_nonmember[:locality_length])
        locality_response = tokenizer.decode(tokenized_nonmember[locality_length:])

        results['prompt'].append(prompt)
        results['original_response'].append(original_response)
        results['target_new'].append(target_new)
        results['generality_prompt'].append(generality_prompt)
        results['locality_prompt'].append(locality_prompt)
        results['locality_response'].append(locality_response)

    return Dataset.from_dict(results)


def build_editing_dataset_w_synthesis(raw_dataset, prompt_ratio=0.5, max_length=1024):
    """
    Build editing dataset using OpenAI-based synthesis.

    Returns dataset with:
    - prompt: First part of member text
    - original_response: Second part of member text
    - target_new: OpenAI-paraphrased response
    - generality_prompt: Same as prompt
    - locality_prompt: First part of non-member text
    - locality_response: Second part of non-member text
    """
    client = OpenAIPrompter(
        api_key=os.getenv('OPENAI_API_KEY'),
        model_name='gpt-4o-mini-2024-07-18'
    )

    results = {
        'prompt': [],
        'original_response': [],
        'target_new': [],
        'generality_prompt': [],
        'locality_prompt': [],
        'locality_response': []
    }

    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-6.9b')

    for member, nonmember, member_neighbors, nonmember_neighbors in zip(
        tqdm(raw_dataset['member'], desc='Processing Editing Dataset...'),
        raw_dataset['nonmember'],
        raw_dataset['member_neighbors'],
        raw_dataset['nonmember_neighbors']
    ):
        # prompt, original_response are split from member
        tokenized_member = tokenizer.encode(member, add_special_tokens=False, truncation=True, max_length=max_length)
        prompt_length = int(prompt_ratio * len(tokenized_member))
        prompt = tokenizer.decode(tokenized_member[:prompt_length])
        original_response = tokenizer.decode(tokenized_member[prompt_length:])

        # target_new is synthesized from the original response
        model_response = client.query(original_response)
        target_new = OpenAIPrompter.parse_response(model_response)

        generality_prompt = prompt

        tokenized_nonmember = tokenizer.encode(nonmember, add_special_tokens=False, truncation=True, max_length=max_length)
        locality_length = int(prompt_ratio * len(tokenized_nonmember))
        locality_prompt = tokenizer.decode(tokenized_nonmember[:locality_length])
        locality_response = tokenizer.decode(tokenized_nonmember[locality_length:])

        results['prompt'].append(prompt)
        results['original_response'].append(original_response)
        results['target_new'].append(target_new)
        results['generality_prompt'].append(generality_prompt)
        results['locality_prompt'].append(locality_prompt)
        results['locality_response'].append(locality_response)

    return Dataset.from_dict(results)


if __name__ == '__main__':
    # Example usage: Generate obfuscated dataset
    raw_dataset = load_raw_dataset('iamgroot42/mimir', 'arxiv', 'ngram_13_0.8', max_samples=10)

    prompt_ratio = 0.5
    editing_dataset = build_editing_dataset_w_synthesis(raw_dataset, prompt_ratio=prompt_ratio, max_length=512)

    dataset_name = f'editing-syn-pr{prompt_ratio}-mimir-arxiv-ngram_13_0.8'
    editing_dataset.save_to_disk(dataset_name)

    # Optional: Push to Hugging Face Hub
    # editing_dataset.push_to_hub(f'X/{dataset_name}', private=False)

    print(f"Dataset saved to {dataset_name}")
    print("To load later:")
    print(f"  Local: load_from_disk('{dataset_name}')")
    print(f"  HF Hub: load_dataset('LLM-MIA/{dataset_name}')")