import torch
from torch.utils.data import Dataset
# Chỉ cần RobertaTokenizerFast nếu AutoTokenizer không hoạt động
from transformers import AutoModel, RobertaTokenizerFast
import json

# ==========================================================
# 1. SỬA ĐỔI: Thêm pad_token_id vào collate_fn
# ==========================================================
def phobert_collate_fn(batch):
    # Lấy pad_token_id từ dataset (đã được lưu trong batch)
    # Giả định: Tất cả các mẫu trong batch đều có cùng pad_token_id
    pad_token_id = batch[0]["pad_token_id"]
    
    input_ids = [torch.tensor(x["input_ids"]) for x in batch]
    attention_mask = [torch.tensor(x["attention_mask"]) for x in batch]
    labels = [torch.tensor(x["labels"]) for x in batch]

    # Dùng pad_token_id chính xác cho input_ids
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=pad_token_id
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask, batch_first=True, padding_value=0
    )
    # Labels vẫn dùng -100, việc xử lý -100 sẽ diễn ra trong hàm train_one_epoch
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
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))

        self.tag2idx = tag2idx
        
        # SỬ DỤNG RobertaTokenizerFast TRỰC TIẾP và thêm add_prefix_space=True
        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            "vinai/phobert-base-v2",
            add_prefix_space=True,
            use_fast=True
        )
        self.max_length = max_length
        
        # 2. SỬA ĐỔI: Lưu ID padding để truyền vào collate_fn
        self.pad_token_id = self.tokenizer.pad_token_id # Thường là 1
        self.cls_token_id = self.tokenizer.cls_token_id # Thường là 0

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
                # Gán -100 cho [CLS], [SEP] và token padding (nếu có)
                labels.append(-100)
            elif word_id != prev_word_id:
                # Gán nhãn cho token đầu tiên của mỗi từ
                labels.append(self.tag2idx[tags[word_id]])
            else:
                # Gán -100 cho các subword (token còn lại) của từ đó
                labels.append(-100)
            prev_word_id = word_id

        encoding["labels"] = labels
        
        # SỬA ĐỔI: Thêm ID padding vào kết quả của __getitem__ để collate_fn sử dụng
        encoding["pad_token_id"] = self.pad_token_id
        
        return encoding
