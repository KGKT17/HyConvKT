import torch
import torch.nn as nn
import torch.nn.functional as F

from .hypergraph import InteractionHyperedgeEncoder


class HyConvKT(nn.Module):
    """HyConvKT: Hypergraph Convolutional Knowledge Tracing.

    Full model combining hypergraph link prediction encoding with
    context-aware dynamic knowledge state evolution.
    """

    def __init__(self, num_questions, num_concepts, embedding_dim=64,
                 d1=8, d2=8, num_difficulties=100, num_types=2,
                 hyperedge_nodes=6, kernel_size=(3, 3, 3),
                 conv_channels=32, dropout=0.2, device=None):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.embedding_dim = embedding_dim

        self.encoder = InteractionHyperedgeEncoder(
            num_questions=num_questions,
            num_concepts=num_concepts,
            embedding_dim=embedding_dim,
            d1=d1,
            d2=d2,
            num_difficulties=num_difficulties,
            num_types=num_types,
            hyperedge_nodes=hyperedge_nodes,
            kernel_size=kernel_size,
            conv_channels=conv_channels,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

        input_dim_gates = embedding_dim * 2 + 1
        self.W_f = nn.Linear(input_dim_gates, embedding_dim)
        self.W_i = nn.Linear(embedding_dim * 2, embedding_dim)
        self.W_c = nn.Linear(embedding_dim * 2, embedding_dim)
        self.ln_cell = nn.LayerNorm(embedding_dim)

        self.pred_fc1 = nn.Linear(embedding_dim, 128)
        self.pred_fc2 = nn.Linear(128, 64)
        self.pred_out = nn.Linear(64, 1)

        for m in [self.W_f, self.W_i, self.W_c,
                  self.pred_fc1, self.pred_fc2, self.pred_out]:
            nn.init.xavier_uniform_(m.weight)

    def forward(self, q_seq, c_seq, r_seq, delta_t_seq, type_seq, diff_seq):
        batch_size, seq_len = q_seq.size()
        h_t = torch.zeros(batch_size, self.embedding_dim, device=self.device)
        logits = []

        for t in range(seq_len):
            q, c = q_seq[:, t], c_seq[:, t]
            diff, typ = diff_seq[:, t], type_seq[:, t]
            dt = delta_t_seq[:, t]
            r = r_seq[:, t]

            query_feat = self.encoder(q, c, diff, typ, dt, r=None, student_state=h_t)

            x = F.relu(self.pred_fc1(query_feat))
            x = self.dropout(x)
            x = F.relu(self.pred_fc2(x))
            x = self.dropout(x)
            logit = self.pred_out(x)
            logits.append(logit)

            interaction_feat = self.encoder(q, c, diff, typ, dt, r=r, student_state=h_t)

            dt_feat = torch.clamp(dt.unsqueeze(-1), -10, 10)
            f_in = torch.cat([h_t, interaction_feat, dt_feat], dim=-1)
            f_t = torch.sigmoid(self.W_f(f_in))
            h_decayed = f_t * h_t

            update_in = torch.cat([h_decayed, interaction_feat], dim=-1)
            i_t = torch.sigmoid(self.W_i(update_in))
            c_t = torch.tanh(self.W_c(update_in))
            c_t = self.dropout(c_t)

            h_t = (1 - i_t) * h_decayed + i_t * c_t
            h_t = self.ln_cell(h_t)

        return torch.stack(logits, dim=1).squeeze(-1)
