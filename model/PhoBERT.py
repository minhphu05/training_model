import torch
from torch import nn
from transformers import AutoModel
from torchcrf import CRF

class PhoBERT_CRF(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.phobert.config.hidden_size, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None, mask=None):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        x = self.dropout(outputs.last_hidden_state)  # (batch, seq_len, hidden)
        emissions = self.fc(x)  # (batch, seq_len, num_tags)

        if labels is not None:
            # mask: True ở vị trí token hợp lệ (không phải padding)
            mask = attention_mask.bool() if mask is None else mask.bool()
            loss = -self.crf(emissions, labels, mask=mask)
            return loss
        else:
            # decode tags
            mask = attention_mask.bool() if mask is None else mask.bool()
            return self.crf.decode(emissions, mask=mask)
