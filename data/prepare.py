from datasets import Dataset, load_dataset


def dataset_prepare(args, tokenizer):
    temp_dataset_name = "arxiv"
    train_end_idx = getattr(args, 'train_end_idx', -1) if args else -1

    raw_dataset = load_dataset("iamgroot42/mimir", name=temp_dataset_name, split='ngram_13_0.8', trust_remote_code=True)
    members = raw_dataset['member']

    if train_end_idx > 0:
        members = members[:train_end_idx]
        print(f"Limited dataset to {train_end_idx} samples")

    print(f"Number of members: {len(members)}")
    members_list = list(members)

    dataset_para = load_dataset(f"LLM-MIA/editing-syn-pr0.5-mimir-{temp_dataset_name}-ngram_13_0.8", split='train')
    para_members_list = [dataset_para[i]['target_new'] for i in range(len(members_list))]
    print("Number of paraphrased members:", len(para_members_list))

    dataset = Dataset.from_dict({
        'member': members_list,
        'member_para': para_members_list
    })

    def tokenize_function(examples):
        entire_text = examples["member"] + examples["member_para"]
        entire_output = tokenizer(
            entire_text,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors=None,
        )
        raw_output_input_ids = entire_output['input_ids'][:len(examples["member"])]
        raw_output_attention_mask = entire_output['attention_mask'][:len(examples["member"])]
        para_output_input_ids = entire_output['input_ids'][len(examples["member"]):]
        para_output_attention_mask = entire_output['attention_mask'][len(examples["member"]):]

        return {
            'input_ids': raw_output_input_ids,
            'attention_mask': raw_output_attention_mask,
            'para_input_ids': para_output_input_ids,
            'para_attention_mask': para_output_attention_mask,
        }

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )

    train_test = tokenized_dataset.train_test_split(test_size=0.1, seed=42)

    print(f"Training samples: {len(train_test['train'])}")
    print(f"Validation samples: {len(train_test['test'])}")

    return train_test['train'], train_test['test']


if __name__ == "__main__":
    from transformers import AutoTokenizer

    model_name = "meta-llama/Llama-3.2-3B"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    train, test = dataset_prepare(None, tokenizer)
    print(f"Sample keys: {train[0].keys()}")