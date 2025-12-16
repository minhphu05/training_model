import torch
from torch.utils.data import Dataset
# Sửa import để lấy RobertaTokenizerFast trực tiếp
from transformers import AutoModel, RobertaTokenizerFast 
import json

# ==========================================================
# phobert_collate_fn (SỬA ĐỔI)
# ==========================================================
def phobert_collate_fn(batch):
    # Lấy pad_token_id từ batch (đã được thêm vào __getitem__)
    pad_token_id = batch[0]["pad_token_id"]
    
    input_ids = [torch.tensor(x["input_ids"]) for x in batch]
    attention_mask = [torch.tensor(x["attention_mask"]) for x in batch]
    labels = [torch.tensor(x["labels"]) for x in batch]

    # Dùng pad_token_id chính xác cho input_ids để tránh lỗi CUDA Indexing
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=pad_token_id
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


# ==========================================================
# PhoBERTNERDataset (SỬA ĐỔI)
# ==========================================================
class PhoBERTNERDataset(Dataset):
    def __init__(self, filepath, tag2idx, max_length=256):
        self.data = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))

        self.tag2idx = tag2idx
        
        # SỬ DỤNG RobertaTokenizerFast TRỰC TIẾP
        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            "vinai/phobert-base-v2",
            add_prefix_space=True, # Quan trọng cho BPE khi is_split_into_words=True
            use_fast=True
        )
        self.max_length = max_length
        
        # Lấy ID padding để truyền vào collate_fn
        self.pad_token_id = self.tokenizer.pad_token_id

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

        # Logic tạo nhãn cho NER (chỉ đánh nhãn token đầu tiên của mỗi từ)
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
        # THÊM ID padding vào kết quả của __getitem__
        encoding["pad_token_id"] = self.pad_token_id
        
        return encoding
