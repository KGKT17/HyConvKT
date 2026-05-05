import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import time
import os
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm


# ==========================================
# 1. Configuration
# ==========================================
class Config:
    def __init__(self):
        # Default values (updated dynamically based on data)
        self.num_students = 0
        self.num_questions = 0
        self.num_concepts = 0
        self.num_types = 2
        self.num_difficulties = 100

        # Model Parameters (Optimized for Memory & Performance)
        self.embedding_dim = 64  # Reverted to 64 for memory safety
        self.d1 = 8  # 8 * 8 = 64
        self.d2 = 8
        self.seq_len = 200
        self.batch_size = 64  # Reduced to 32 to prevent OOM with 3D Convs
        self.learning_rate = 5e-4
        self.epochs = 200
        self.dropout = 0.3  # Moderate dropout
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 3D Convolution Parameters
        self.kernel_size = (3, 3, 3)
        self.hyperedge_nodes = 6  # u, q, c, d, type, time

        # --- Saving & Loading ---
        self.save_dir = "./checkpoints"
        self.model_name = "hglp_kt_pykt_attention.pth"
        self.resume = True

        # --- Data Configuration ---
        self.data_dir = "./data/xes3g5m"
        self.file_name = "train_valid_sequences.csv"
        self.fold = 0
        self.patience = 20


config = Config()


# ==========================================
# 2. Data Processing & Loading
# ==========================================

def get_difficulty_map(df):
    """Calculate question difficulty (1 - accuracy)"""
    print("Calculating question difficulties...")
    q_dict = {}  # {qid: [correct_count, total_count]}

    for index, row in tqdm(df.iterrows(), total = df.shape[0], desc = "Calc Difficulty"):
        q_seq = row['questions']
        r_seq = row['responses']
        valid_indices = [i for i, q in enumerate(q_seq) if q != 0 and r_seq[i] in [0, 1]]

        for i in valid_indices:
            qid = q_seq[i]
            r = r_seq[i]
            if qid not in q_dict:
                q_dict[qid] = [0, 0]
            q_dict[qid][1] += 1
            if r == 1:
                q_dict[qid][0] += 1

    diff_map = {}
    for qid, (correct, total) in q_dict.items():
        acc = correct / total if total > 0 else 0.5
        diff = int((1 - acc) * 100) + 1
        diff_map[qid] = diff

    return diff_map


def load_pykt_data(config):
    """Load and parse CSV data"""
    file_path = os.path.join(config.data_dir, config.file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    print(f"Loading CSV: {file_path}")
    df = pd.read_csv(file_path)

    print("Parsing string sequences...")
    for col in ['questions', 'concepts', 'responses', 'timestamps']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: list(map(int, str(x).split(','))) if pd.notnull(x) else [])

    train_df = df[df['fold'] != config.fold].copy()
    valid_df = df[df['fold'] == config.fold].copy()

    print(f"Split Complete - Train: {len(train_df)}, Valid (Fold {config.fold}): {len(valid_df)}")

    diff_map = get_difficulty_map(train_df)

    def df_to_list(dataframe):
        data_list = []
        for _, row in dataframe.iterrows():
            item = {
                'uid': row['uid'],
                'q_seq': row['questions'],
                'c_seq': row['concepts'],
                'r_seq': row['responses'],
                't_seq': row['timestamps'] if 'timestamps' in row else [0] * len(row['questions'])
            }
            data_list.append(item)
        return data_list

    train_data = df_to_list(train_df)
    valid_data = df_to_list(valid_df)

    return train_data, valid_data, diff_map


class PyKT_Dataset(Dataset):
    def __init__(self, data, diff_map, config):
        self.data = data
        self.diff_map = diff_map
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]

        q_seq = row['q_seq']
        diff_seq = [self.diff_map.get(q, 50) for q in q_seq]
        type_seq = [1] * len(q_seq)

        q_tensor = torch.tensor(q_seq, dtype = torch.long)
        c_tensor = torch.tensor(row['c_seq'], dtype = torch.long)
        r_tensor = torch.tensor(row['r_seq'], dtype = torch.long)
        t_tensor = torch.tensor(row['t_seq'], dtype = torch.long)
        type_tensor = torch.tensor(type_seq, dtype = torch.long)
        diff_tensor = torch.tensor(diff_seq, dtype = torch.long)

        delta_t = torch.zeros_like(t_tensor, dtype = torch.float)
        if len(t_tensor) > 1:
            raw_delta = (t_tensor[1:] - t_tensor[:-1]).float()
            raw_delta = torch.abs(raw_delta)
            delta_t[1:] = torch.log(raw_delta / 60.0 + 1.0)

        return {
            'q_seq': q_tensor,
            'c_seq': c_tensor,
            'r_seq': r_tensor,
            'delta_t': delta_t,
            'type_seq': type_tensor,
            'diff_seq': diff_tensor,
            'uid': row['uid']
        }


def collate_fn(batch):
    batch_data = {}
    r_seqs = [x['r_seq'] for x in batch]
    batch_data['r_seq'] = pad_sequence(r_seqs, batch_first = True, padding_value = -1)

    for key in ['q_seq', 'c_seq', 'type_seq', 'diff_seq']:
        seqs = [x[key] for x in batch]
        seqs = [s[-(config.seq_len):] for s in seqs]
        batch_data[key] = pad_sequence(seqs, batch_first = True, padding_value = 0)

    dt_seqs = [x['delta_t'] for x in batch]
    dt_seqs = [s[-(config.seq_len):] for s in dt_seqs]
    batch_data['delta_t'] = pad_sequence(dt_seqs, batch_first = True, padding_value = 0)

    if batch_data['r_seq'].size(1) > config.seq_len:
        batch_data['r_seq'] = batch_data['r_seq'][:, -config.seq_len:]

    return batch_data


# The code will be shared after published.


# ==========================================
# 4. Save/Load Utils
# ==========================================

def save_checkpoint(model, optimizer, epoch, best_auc, config):
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    save_path = os.path.join(config.save_dir, config.model_name)
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_auc': best_auc,
        'config': {'embedding_dim': config.embedding_dim}
    }
    torch.save(state, save_path)
    print(f"[*] Model Saved: {save_path} (Epoch: {epoch + 1}, AUC: {best_auc:.4f})")


def load_checkpoint(model, optimizer, config):
    load_path = os.path.join(config.save_dir, config.model_name)
    if not os.path.exists(load_path):
        print(f"[!] Checkpoint not found: {load_path}. Starting fresh.")
        return 0, 0.5

    print(f"[*] Loading Checkpoint: {load_path} ...")
    checkpoint = torch.load(load_path, map_location = config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_auc = checkpoint['best_auc']
    print(f"[*] Loaded. Resume from Epoch {start_epoch + 1} (Best AUC: {best_auc:.4f})")
    return start_epoch, best_auc


# ==========================================
# 5. Training & Evaluation
# ==========================================

def train_model(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    valid_batches = 0
    loop = tqdm(dataloader, desc = "Training", leave = False)

    for batch in loop:
        q = torch.clamp(batch['q_seq'], 0, config.num_questions + 9).to(config.device)
        c = torch.clamp(batch['c_seq'], 0, config.num_concepts + 9).to(config.device)
        diff = torch.clamp(batch['diff_seq'], 0, config.num_difficulties + 9).to(config.device)
        typ = torch.clamp(batch['type_seq'], 0, config.num_types + 9).to(config.device)
        dt = batch['delta_t'].to(config.device)
        r = batch['r_seq'].to(config.device)

        optimizer.zero_grad()
        logits = model(q, c, r, dt, typ, diff)

        mask = (q != 0) & (r >= 0) & (r <= 1)
        target = r.float()
        target = torch.where(mask, target, torch.zeros_like(target))

        loss_elementwise = criterion(logits, target)

        mask_sum = mask.sum()
        if mask_sum > 0:
            loss = (loss_elementwise * mask).sum() / mask_sum
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
            optimizer.step()

            total_loss += loss.item()
            valid_batches += 1
            loop.set_postfix(loss = loss.item())

    return total_loss / max(valid_batches, 1)


def evaluate_model(model, dataloader, criterion):
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0
    loop = tqdm(dataloader, desc = "Evaluating", leave = False)

    with torch.no_grad():
        for batch in loop:
            q = torch.clamp(batch['q_seq'], 0, config.num_questions + 9).to(config.device)
            c = torch.clamp(batch['c_seq'], 0, config.num_concepts + 9).to(config.device)
            diff = torch.clamp(batch['diff_seq'], 0, config.num_difficulties + 9).to(config.device)
            typ = torch.clamp(batch['type_seq'], 0, config.num_types + 9).to(config.device)
            dt = batch['delta_t'].to(config.device)
            r = batch['r_seq'].to(config.device)

            logits = model(q, c, r, dt, typ, diff)
            mask = (q != 0) & (r >= 0) & (r <= 1)
            target = torch.where(mask, r.float(), torch.zeros_like(r.float()))

            loss = (criterion(logits, target) * mask).sum() / (mask.sum() + 1e-8)
            total_loss += loss.item()

            active_elements = mask.bool()
            y_true.extend(target[active_elements].cpu().numpy())
            y_pred.extend(torch.sigmoid(logits[active_elements]).cpu().numpy())

    try:
        auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.5
    except:
        auc = 0.5
    acc = accuracy_score(y_true, [1 if p > 0.5 else 0 for p in y_pred])

    return total_loss / len(dataloader), auc, acc


# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    print(f"Using Device: {config.device}")

    # 1. Load Data
    try:
        train_data, valid_data, diff_map = load_pykt_data(config)
    except Exception as e:
        print(f"Data loading failed: {e}")
        exit()

    print(f"Data Loaded. Train: {len(train_data)}, Valid: {len(valid_data)}")

    # 2. Update Config
    print("Updating Config...")
    max_q, max_c = 0, 0
    for row in train_data + valid_data:
        if len(row['q_seq']) > 0:
            max_q = max(max_q, max(row['q_seq']))
        if len(row['c_seq']) > 0:
            max_c = max(max_c, max(row['c_seq']))

    config.num_questions = max_q + 1
    config.num_concepts = max_c + 1
    print(f"Config Updated: Num Q: {config.num_questions}, Num C: {config.num_concepts}")

    # 3. Create DataLoaders
    train_dataset = PyKT_Dataset(train_data, diff_map, config)
    valid_dataset = PyKT_Dataset(valid_data, diff_map, config)

    train_loader = DataLoader(train_dataset, batch_size = config.batch_size, shuffle = True, collate_fn = collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size = config.batch_size, shuffle = False, collate_fn = collate_fn)

    # 4. Initialize Model
    model = HGLP_KT(config).to(config.device)

    # Optimizer with Weight Decay
    optimizer = torch.optim.Adam(model.parameters(), lr = config.learning_rate, weight_decay = 1e-4)

    criterion = nn.BCEWithLogitsLoss(reduction = 'none')

    # 5. Resume logic
    start_epoch = 0
    best_auc = 0.5
    if config.resume:
        start_epoch, best_auc = load_checkpoint(model, optimizer, config)

    # 6. Training Loop
    print(f"Start Training (Epoch {start_epoch + 1} to {config.epochs})...")
    patience_counter = 0

    for epoch in range(start_epoch, config.epochs):
        start_time = time.time()

        train_loss = train_model(model, train_loader, optimizer, criterion)
        val_loss, val_auc, val_acc = evaluate_model(model, valid_loader, criterion)

        epoch_time = time.time() - start_time

        print(f"Epoch {epoch + 1}/{config.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val AUC: {val_auc:.4f} | "
              f"Val ACC: {val_acc:.4f} | "
              f"Time: {epoch_time:.1f}s")

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, best_auc, config)
            print(f"    >>> New Best AUC! Model Saved.")
        else:
            patience_counter += 1
            print(f"    >>> No improvement. Patience: {patience_counter}/{config.patience}")

            if patience_counter >= config.patience:
                print(f"\n[!] Early Stopping triggered after {config.patience} epochs.")
                print(f"[*] Best Validation AUC: {best_auc:.4f}")
                break
