import torch
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer
import json

def phobert_collate_fn(batch):
    input_ids = [torch.tensor(x["input_ids"]) for x in batch]
    attention_mask = [torch.tensor(x["attention_mask"]) for x in batch]
    labels = [torch.tensor(x["labels"]) for x in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=1
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask, batch_first=True, padding_value=0
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


class PhoBERTNERDataset(Dataset):
    def __init__(self, filepath, tag2idx, max_length=256):
        self.data = []
        # phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))

        self.tag2idx = tag2idx
        self.tokenizer = AutoTokenizer.from_pretrained(
            "vinai/phobert-base-v2",
            use_fast=True
        )
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        words = item["tokens"]
        tags = item["tags"]

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors=None
        )

        word_ids = encoding.word_ids()
        labels = []

        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)
            elif word_id != prev_word_id:
                labels.append(self.tag2idx[tags[word_id]])
            else:
                labels.append(-100)
            prev_word_id = word_id

        encoding["labels"] = labels
        return encoding
