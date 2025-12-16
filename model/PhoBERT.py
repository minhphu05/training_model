import torch
from torch import nn
from transformers import AutoModel
from torchcrf import CRF

class PhoBERT_CRF(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(768, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None, mask=None): 
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        x = self.dropout(outputs.last_hidden_state)
        emissions = self.fc(x)

        if labels is not None:
            # 2. SỬ DỤNG MASK TRỰC TIẾP
            # Mask đã được xử lý (lọc -100) ở hàm train_one_epoch.
            # Ta cần chuyển mask từ Long Tensor sang Boolean Tensor.
            crf_mask = mask.bool() if mask is not None else attention_mask.bool()
            
            loss = -self.crf(
                emissions,
                labels,
                mask=crf_mask # <--- SỬ DỤNG MASK LỌC CẢ -100 VÀ PADDING
            )
            return loss
        else:
            # 3. SỬ DỤNG attention_mask mặc định cho decode (lọc padding)
            return self.crf.decode(
                emissions,
                mask=attention_mask.bool()
            )
