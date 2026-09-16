import torch
import torch.nn as nn
import torch.nn.functional as F


class NodeAttention(nn.Module):
    """Attention aggregation over 6 hyperedge nodes (Student, Q, C, D, Type, Time)."""

    def __init__(self, channels, d1, d2):
        super().__init__()
        self.dim = channels * d1 * d2
        self.attn = nn.Sequential(
            nn.Linear(self.dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1, bias=False),
        )

    def forward(self, x):
        B, C, D, H, W = x.size()
        x_flat = x.permute(0, 2, 1, 3, 4).reshape(B, D, -1)
        scores = self.attn(x_flat)
        weights = F.softmax(scores, dim=1)
        out = (x_flat * weights).sum(dim=1)
        return out


class InteractionHyperedgeEncoder(nn.Module):
    """Encodes a single interaction hyperedge via 3D circular convolution."""

    def __init__(self, num_questions, num_concepts, embedding_dim, d1, d2,
                 num_difficulties=100, num_types=2, hyperedge_nodes=6,
                 kernel_size=(3, 3, 3), conv_channels=32, dropout=0.2):
        super().__init__()
        self.d1, self.d2 = d1, d2
        assert d1 * d2 == embedding_dim

        self.q_emb = nn.Embedding(num_questions + 10, embedding_dim, padding_idx=0)
        self.c_emb = nn.Embedding(num_concepts + 10, embedding_dim, padding_idx=0)
        self.diff_emb = nn.Embedding(num_difficulties + 10, embedding_dim, padding_idx=0)
        self.type_emb = nn.Embedding(num_types + 10, embedding_dim, padding_idx=0)
        self.time_proj = nn.Linear(1, embedding_dim)
        self.response_emb = nn.Embedding(3, embedding_dim, padding_idx=2)

        self.emb_dropout = nn.Dropout(dropout)
        self.pos_emb = nn.Parameter(torch.randn(1, 1, hyperedge_nodes, d1, d2) * 0.02)

        from .conv3d import CircularConv3d, SEBlock3D
        self.conv1 = CircularConv3d(1, 16, kernel_size=kernel_size)
        self.conv2 = CircularConv3d(16, conv_channels, kernel_size=kernel_size)
        self.se = SEBlock3D(conv_channels, reduction=4)
        self.node_attn = NodeAttention(conv_channels, d1, d2)

        self.fc = nn.Linear(conv_channels * d1 * d2, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)

        nn.init.xavier_uniform_(self.time_proj.weight)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, q, c, diff, typ, delta_t, r=None, student_state=None):
        batch_size = q.size(0)

        e_q = self.q_emb(q)
        e_c = self.c_emb(c)
        e_diff = self.diff_emb(diff)
        e_type = self.type_emb(typ)

        dt_safe = torch.clamp(delta_t.unsqueeze(-1), min=-10, max=10)
        e_time = self.time_proj(dt_safe)

        if r is not None:
            r_idx = r.clone()
            r_idx[r_idx < 0] = 2
            r_idx[r_idx > 1] = 2
            e_r = self.response_emb(r_idx)
            e_q = e_q + e_r

        if student_state is None:
            student_state = torch.zeros_like(e_q)

        stack_list = [student_state, e_q, e_c, e_diff, e_type, e_time]
        reshaped_list = [x.view(batch_size, 1, self.d1, self.d2) for x in stack_list]
        X_raw = torch.cat(reshaped_list, dim=1)

        X_in = X_raw.unsqueeze(1)
        X_in = X_in + self.pos_emb
        X_in = self.emb_dropout(X_in)

        out = F.relu(self.conv1(X_in))
        out = F.relu(self.conv2(out))
        out = self.se(out)
        out = self.node_attn(out)

        out = self.fc(out)
        return self.layer_norm(out)
