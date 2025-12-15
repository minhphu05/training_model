import torch
from torch import nn
from torchcrf import CRF

class LSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_tags: int,
        embedding_dim: int,
        num_layers: int = 2,
        hidden_size: int = 256,
        padding_idx: int = 0,
        dropout: float = 0.3,
        **kwargs: any
    ):
        super().__init()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        self.dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(
            in_features=hidden_size * 2,
            out_features=num_tags
        )
        self.crf = CRF(
            num_tags=num_tags,
            batch_first=True
        )
    
    def forward(
      self,
      input_ids: torch.Tensor,
      length: torch.Tensor  
    ) ->None: 
        