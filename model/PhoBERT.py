import torch
from torch import nn
from transformers import AutoModel
from torchcrf import CRF

class PhoBERT_CRF(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        # Cảnh báo "You should probably TRAIN..." là bình thường
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(768, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    # Thêm tham số mask=None
    def forward(self, input_ids, attention_mask, labels=None, mask=None): 
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        x = self.dropout(outputs.last_hidden_state)
        emissions = self.fc(x) # (BatchSize, SeqLen, NumTags)

        if labels is not None:
            # Mask đã được xử lý (lọc -100 và sửa lỗi mask[:, 0]) ở hàm train_one_epoch
            # Ta chỉ cần sử dụng nó.
            crf_mask = mask.bool() if mask is not None else attention_mask.bool()
            
            # Tính toán Negative Log-Likelihood Loss (âm vì CRF loss là negative log-likelihood)
            loss = -self.crf(
                emissions,
                labels,
                mask=crf_mask
            )
            return loss
        else:
            # Decode (dự đoán) - chỉ cần lọc padding bằng attention_mask
            return self.crf.decode(
                emissions,
                mask=attention_mask.bool()
            )
