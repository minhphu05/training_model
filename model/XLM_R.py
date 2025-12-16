import torch
from torch import nn
from transformers import AutoModel
from torchcrf import CRF

class XLMR_CRF(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        # Thay đổi model từ phobert sang xlm-roberta-base
        self.xlmroberta = AutoModel.from_pretrained("xlm-roberta-base")
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(768, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.xlmroberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        x = self.dropout(outputs.last_hidden_state)
        emissions = self.fc(x)

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
