import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm


class Trainer:
    """Training engine for HyConvKT."""

    def __init__(self, model, optimizer, criterion, config, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.device = device

    def train_epoch(self, dataloader, num_questions, num_concepts, num_difficulties, num_types):
        self.model.train()
        total_loss = 0
        valid_batches = 0
        loop = tqdm(dataloader, desc="Training", leave=False)

        for batch in loop:
            q = torch.clamp(batch['q_seq'], 0, num_questions + 9).to(self.device)
            c = torch.clamp(batch['c_seq'], 0, num_concepts + 9).to(self.device)
            diff = torch.clamp(batch['diff_seq'], 0, num_difficulties + 9).to(self.device)
            typ = torch.clamp(batch['type_seq'], 0, num_types + 9).to(self.device)
            dt = batch['delta_t'].to(self.device)
            r = batch['r_seq'].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(q, c, r, dt, typ, diff)

            mask = (q != 0) & (r >= 0) & (r <= 1)
            target = r.float()
            target = torch.where(mask, target, torch.zeros_like(target))

            loss_elementwise = self.criterion(logits, target)
            mask_sum = mask.sum()

            if mask_sum > 0:
                loss = (loss_elementwise * mask).sum() / mask_sum
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.config['training']['grad_clip']
                )
                self.optimizer.step()

                total_loss += loss.item()
                valid_batches += 1
                loop.set_postfix(loss=loss.item())

        return total_loss / max(valid_batches, 1)

    def fit(self, train_loader, val_loader, num_questions, num_concepts,
            num_difficulties, num_types, evaluator):
        save_dir = self.config['checkpoint']['save_dir']
        model_name = self.config['checkpoint']['model_name']
        os.makedirs(save_dir, exist_ok=True)

        best_auc = 0.5
        patience_counter = 0
        patience = self.config['training']['patience']
        epochs = self.config['training']['epochs']

        for epoch in range(epochs):
            start_time = time.time()

            train_loss = self.train_epoch(
                train_loader, num_questions, num_concepts, num_difficulties, num_types
            )
            val_loss, val_auc, val_acc = evaluator.evaluate(
                val_loader, num_questions, num_concepts, num_difficulties, num_types
            )

            epoch_time = time.time() - start_time
            print(f"Epoch {epoch + 1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val AUC: {val_auc:.4f} | "
                  f"Val ACC: {val_acc:.4f} | "
                  f"Time: {epoch_time:.1f}s")

            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                save_path = os.path.join(save_dir, model_name)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_auc': best_auc,
                }, save_path)
                print(f"    >>> New Best AUC! Model saved.")
            else:
                patience_counter += 1
                print(f"    >>> No improvement. Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print(f"\n[!] Early stopping at epoch {epoch + 1}.")
                    break

        return best_auc
