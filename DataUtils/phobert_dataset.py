import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizerFast
import json

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

class phoNERTNER(Dataset):
    def __init__(self, filepath, tokenizer):
        self.tokenizer = tokenizer
        self.data = read_json_or_jsonl(filepath)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens, tags = extract_data(item)  # tokens: list of words, tags: list of tags per word

        encoding = self.tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True)
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        word_ids = encoding.word_ids()
        new_tags = []
        for word_idx in word_ids:
            if word_idx is None:
                new_tags.append(-100)  # padding or special tokens
            else:
                new_tags.append(vocab.tag2idx.get(tags[word_idx], -100))

        tags_ids = torch.tensor(new_tags, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "tags_ids": tags_ids
        }


def phobert_collate_fn(batch):
    pad_token_id = batch[0]["pad_token_id"]

    input_ids = [torch.tensor(x["input_ids"]) for x in batch]
    attention_mask = [torch.tensor(x["attention_mask"]) for x in batch]
    labels = [torch.tensor(x["labels"]) for x in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
