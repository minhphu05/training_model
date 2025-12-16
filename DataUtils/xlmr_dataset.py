import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class NERDataset(Dataset):
    def __init__(self, data, label2id, max_len=128):
        """
        data: list dict, mỗi dict có "tokens": [...], "tags": [...]
        label2id: dict map tag string -> int
        max_len: int max length câu
        """
        self.data = data
        self.label2id = label2id
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]["tokens"]
        tags = self.data[idx]["tags"]

        # Tokenize giữ nguyên tokens (is_split_into_words=True)
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_offsets_mapping=False,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze()  # (max_len,)
        attention_mask = encoding["attention_mask"].squeeze()  # (max_len,)

        # Map tags theo subtoken
        word_ids = encoding.word_ids(batch_index=0)  # list[int or None]

        labels = []
        for word_idx in word_ids:
            if word_idx is None:
                labels.append(self.label2id["O"])  # hoặc 0, nhãn mặc định cho padding
            else:
                label = tags[word_idx]
                labels.append(self.label2id[label])

        # Padding nếu nhãn ngắn hơn max_len
        if len(labels) < self.max_len:
            labels.extend([-100] * (self.max_len - len(labels)))
        else:
            labels = labels[:self.max_len]

        labels = torch.tensor(labels)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }