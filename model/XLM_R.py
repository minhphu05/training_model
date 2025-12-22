import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel

# class XLMR_BiLSTM(nn.Module):
#     def __init__(
#         self,
#         num_tags: int,
#         embedding_dim: int = 768,
#         num_layers: int = 2,
#         hidden_size: int = 256,
#         dropout: float = 0.3
#     ):
#         super().__init__()
#         self.xlmroberta = AutoModel.from_pretrained("xlm-roberta-base")
#         self.dropout = nn.Dropout(dropout)
        
#         self.bilstm = nn.LSTM(
#             input_size=embedding_dim,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True,
#             bidirectional=True,
#             dropout=dropout if num_layers > 1 else 0
#         )
        
#         self.fc = nn.Linear(hidden_size * 2, num_tags)

#     # def forward(self, input_ids, attention_mask, lengths):
#     #     outputs = self.xlmroberta(
#     #         input_ids=input_ids,
#     #         attention_mask=attention_mask
#     #     )
#     #     x = self.dropout(outputs.last_hidden_state)  # (batch, seq_len, 768)
        
#     #     # Xử lý pack padded sequence để bỏ phần padding
#     #     packed_input = pack_padded_sequence(
#     #         x, 
#     #         lengths.cpu(),
#     #         batch_first=True,
#     #         enforce_sorted=False
#     #     )
        
#     #     packed_output, _ = self.bilstm(packed_input)
        
#     #     output, _ = pad_packed_sequence(
#     #         packed_output,
#     #         batch_first=True
#     #     )
        
#     #     output = self.dropout(output)
#     #     logits = self.fc(output)  # (batch, seq_len, num_tags)
#     #     return logits
#     def forward(self, input_ids, attention_mask):
#         outputs = self.xlmroberta(
#             input_ids=input_ids,
#             attention_mask=attention_mask
#         )
#         x = self.dropout(outputs.last_hidden_state)  # (batch, seq_len, 768)
    
#         # Đừng dùng pack_padded_sequence ở đây
#         lstm_out, _ = self.bilstm(x)  # (batch, seq_len, hidden_size*2)
    
#         lstm_out = self.dropout(lstm_out)
#         logits = self.fc(lstm_out)  # (batch, seq_len, num_tags)
#         return logits

# class XLMR_BiLSTM(nn.Module):
#     def __init__(
#         self,
#         num_tags: int,
#         embedding_dim: int = 768,
#         num_layers: int = 2,
#         hidden_size: int = 256,
#         dropout: float = 0.3
#     ):
#         super().__init__()
#         self.xlmroberta = AutoModel.from_pretrained("xlm-roberta-base")
#         self.dropout = nn.Dropout(dropout)

#         self.bilstm = nn.LSTM(
#             input_size=embedding_dim,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True,
#             bidirectional=True,
#             dropout=dropout if num_layers > 1 else 0
#         )

#         self.fc = nn.Linear(hidden_size * 2, num_tags)

#     def forward(self, input_ids, attention_mask, lengths):
#         outputs = self.xlmroberta(
#             input_ids=input_ids,
#             attention_mask=attention_mask
#         )

#         x = self.dropout(outputs.last_hidden_state)

#         packed = pack_padded_sequence(
#             x,
#             lengths.cpu(),
#             batch_first=True,
#             enforce_sorted=False
#         )

#         packed_out, _ = self.bilstm(packed)

#         lstm_out, _ = pad_packed_sequence(
#             packed_out,
#             batch_first=True,
#             total_length=input_ids.size(1)  # ⭐ CỰC KỲ QUAN TRỌNG
#         )

#         logits = self.fc(self.dropout(lstm_out))
#         return logits

import torch
from torch import nn
from transformers import AutoModel
from torchcrf import CRF

class XLMR_CRF(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        # Load xlm-roberta-large
        self.xlmroberta = AutoModel.from_pretrained("xlm-roberta-base")
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(768, num_tags)  # chỉnh 1024 thay vì 768
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

    
