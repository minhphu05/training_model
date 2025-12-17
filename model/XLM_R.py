import torch
from torch import nn
from transformers import AutoModel

class XLMR_BiLSTM(nn.Module):
    def __init__(
        self,
        num_tags: int,
        embedding_dim: int = 768,
        num_layers: int = 2,
        hidden_size: int = 256,
        dropout: float = 0.3
    ):
        super().__init__()
        self.xlmroberta = AutoModel.from_pretrained("xlm-roberta-base")
        self.dropout = nn.Dropout(dropout)
        
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size * 2, num_tags)

    def forward(self, input_ids, attention_mask):
        outputs = self.xlmroberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        x = self.dropout(outputs.last_hidden_state)  # (batch, seq_len, 768)
        
        lstm_out, _ = self.bilstm(x)  # (batch, seq_len, hidden_size*2)
        
        logits = self.fc(lstm_out)  # (batch, seq_len, num_tags)
        return logits
