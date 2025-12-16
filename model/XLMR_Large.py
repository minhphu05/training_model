import torch
from torch import nn
from transformers import AutoModel
from torchcrf import CRF

class XLMR_CRF(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        # Load xlm-roberta-large
        self.xlmroberta = AutoModel.from_pretrained("xlm-roberta-large")
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(1024, num_tags)  # chỉnh 1024 thay vì 768
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.xlmroberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        x = self.dropout(outputs.last_hidden_state)  # (batch, seq_len, 1024)
        emissions = self.fc(x)  # (batch, seq_len, num_tags)

        if labels is not None:
            loss = -self.crf(
                emissions,
                labels,
                mask=attention_mask.bool()
            )
            return loss
        else:
            return self.crf.decode(
                emissions,
                mask=attention_mask.bool()
            )
