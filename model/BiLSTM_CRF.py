import torch
from torch import nn
from torchcrf import CRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiLSTM_CRF(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_tags: int,
        embedding_dim: int,
        num_layers: int = 2,
        hidden_size: int = 256,
        padding_idx: int = 0,
        dropout: float = 0.3
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )

        self.dropout = nn.Dropout(dropout)

        self.bilstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_size * 2, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        tags: torch.Tensor | None = None
    ):
        mask = input_ids != 0   # padding_idx = 0

        emb = self.dropout(self.embedding(input_ids))

        packed = pack_padded_sequence(
            emb, lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_out, _ = self.bilstm(packed)

        lstm_out, _ = pad_packed_sequence(
            packed_out,
            batch_first=True
        )

        emissions = self.fc(self.dropout(lstm_out))

        # ===== TRAIN =====
        if tags is not None:
            loss = -self.crf(emissions, tags, mask=mask)
            return loss

        # ===== INFER =====
        preds = self.crf.decode(emissions, mask=mask)
        return preds
