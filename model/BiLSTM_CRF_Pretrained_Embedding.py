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
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        padding_idx: int = 0,
        pretrained_embeddings: torch.Tensor | None = None,
        freeze_embedding: bool = False
    ):
        super().__init__()

        # ===== Embedding =====
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx
        )

        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = not freeze_embedding

        self.dropout = nn.Dropout(dropout)

        # ===== BiLSTM Encoder =====
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # ===== Linear projection =====
        self.fc = nn.Linear(hidden_size * 2, num_tags)

        # ===== CRF =====
        self.crf = CRF(num_tags, batch_first=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        tags: torch.Tensor | None = None
    ):
        """
        Train:
            loss = model(input_ids, lengths, tags)
        Eval:
            preds = model(input_ids, lengths)
        """

        # input_ids: [B, T]
        mask = input_ids != 0  # padding_idx = 0

        # ===== Embedding =====
        x = self.embedding(input_ids)          # [B, T, E]
        x = self.dropout(x)

        # ===== Pack sequence =====
        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_out, _ = self.bilstm(packed)

        # ===== Unpack =====
        lstm_out, _ = pad_packed_sequence(
            packed_out,
            batch_first=True
        )  # [B, T, 2H]

        lstm_out = self.dropout(lstm_out)

        # ===== Emissions =====
        emissions = self.fc(lstm_out)  # [B, T, num_tags]

        # ===== TRAIN =====
        if tags is not None:
            # CRF returns log-likelihood → ta MINUS để thành loss
            loss = -self.crf(
                emissions,
                tags,
                mask=mask,
                reduction="mean"
            )
            return loss

        # ===== INFERENCE =====
        else:
            preds = self.crf.decode(
                emissions,
                mask=mask
            )
            return preds
