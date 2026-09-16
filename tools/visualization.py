import torch
import numpy as np
import matplotlib.pyplot as plt


def visualize_knowledge_state(model, sample, concept_names=None, save_path=None):
    """Visualize knowledge state evolution as a heatmap.

    Shows how predicted mastery of each concept changes over
    a sequence of interactions, with correct/incorrect markers.
    """
    model.eval()
    with torch.no_grad():
        q = sample['q_seq'].unsqueeze(0)
        c = sample['c_seq'].unsqueeze(0)
        r = sample['r_seq'].unsqueeze(0)
        dt = sample['delta_t'].unsqueeze(0)
        typ = sample['type_seq'].unsqueeze(0)
        diff = sample['diff_seq'].unsqueeze(0)

        device = next(model.parameters()).device
        q = q.to(device)
        c = c.to(device)
        r = r.to(device)
        dt = dt.to(device)
        typ = typ.to(device)
        diff = diff.to(device)

        batch_size, seq_len = q.size()
        h_t = torch.zeros(batch_size, model.embedding_dim, device=device)

        steps = min(seq_len, 60)
        concept_mastery = {}

        for t in range(steps):
            q_t, c_t = q[:, t], c[:, t]
            r_t = r[:, t]

            query_feat = model.encoder(q_t, c_t, diff[:, t], typ[:, t], dt[:, t],
                                       r=None, student_state=h_t)
            pred = torch.sigmoid(model.pred_out(
                torch.relu(model.pred_fc2(
                    torch.dropout(
                        torch.relu(model.pred_fc1(query_feat)),
                        p=0.2, training=False
                    )
                ))
            )).item()

            concept_id = c_t.item()
            if concept_id not in concept_mastery:
                concept_mastery[concept_id] = []
            concept_mastery[concept_id].append((t, pred, r_t.item()))

            interaction_feat = model.encoder(q_t, c_t, diff[:, t], typ[:, t], dt[:, t],
                                             r=r_t, student_state=h_t)
            dt_feat = torch.clamp(dt[:, t].unsqueeze(-1), -10, 10)
            f_in = torch.cat([h_t, interaction_feat, dt_feat], dim=-1)
            f_t = torch.sigmoid(model.W_f(f_in))
            h_decayed = f_t * h_t
            update_in = torch.cat([h_decayed, interaction_feat], dim=-1)
            i_t = torch.sigmoid(model.W_i(update_in))
            c_t_g = torch.tanh(model.W_c(update_in))
            h_t = (1 - i_t) * h_decayed + i_t * c_t_g
            h_t = model.ln_cell(h_t)

    if not concept_mastery:
        return

    concepts = sorted(concept_mastery.keys())
    max_steps = max(len(v) for v in concept_mastery.values())

    matrix = np.full((len(concepts), max_steps), np.nan)
    for i, cid in enumerate(concepts):
        for step, mastery, _ in concept_mastery[cid]:
            matrix[i, step] = mastery

    fig, ax = plt.subplots(figsize=(14, max(3, len(concepts) * 0.4)))
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1,
                   interpolation='nearest')
    ax.set_xlabel('Interaction Step')
    ax.set_ylabel('Knowledge Concept')
    ax.set_title('Predicted Knowledge Mastery Over Time')
    plt.colorbar(im, ax=ax, label='Mastery Probability')

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    plt.close()
