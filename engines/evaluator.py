import os
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm


class Evaluator:
    """Evaluation engine for HyConvKT."""

    def __init__(self, model, criterion, device):
        self.model = model
        self.criterion = criterion
        self.device = device

    def evaluate(self, dataloader, num_questions, num_concepts,
                 num_difficulties, num_types):
        self.model.eval()
        y_true, y_pred = [], []
        total_loss = 0
        loop = tqdm(dataloader, desc="Evaluating", leave=False)

        with torch.no_grad():
            for batch in loop:
                q = torch.clamp(batch['q_seq'], 0, num_questions + 9).to(self.device)
                c = torch.clamp(batch['c_seq'], 0, num_concepts + 9).to(self.device)
                diff = torch.clamp(batch['diff_seq'], 0, num_difficulties + 9).to(self.device)
                typ = torch.clamp(batch['type_seq'], 0, num_types + 9).to(self.device)
                dt = batch['delta_t'].to(self.device)
                r = batch['r_seq'].to(self.device)

                logits = self.model(q, c, r, dt, typ, diff)
                mask = (q != 0) & (r >= 0) & (r <= 1)
                target = torch.where(mask, r.float(), torch.zeros_like(r.float()))

                loss = (self.criterion(logits, target) * mask).sum() / (mask.sum() + 1e-8)
                total_loss += loss.item()

                active_elements = mask.bool()
                y_true.extend(target[active_elements].cpu().numpy())
                y_pred.extend(torch.sigmoid(logits[active_elements]).cpu().numpy())

        try:
            auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.5
        except Exception:
            auc = 0.5
        acc = accuracy_score(y_true, [1 if p > 0.5 else 0 for p in y_pred])

        return total_loss / max(len(dataloader), 1), auc, acc

    def load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            print(f"[!] Checkpoint not found: {checkpoint_path}")
            return 0, 0.5

        print(f"[*] Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_auc = checkpoint.get('best_auc', 0.5)
        print(f"[*] Loaded. Best AUC: {best_auc:.4f}")
        return start_epoch, best_auc
