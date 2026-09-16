import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


def get_difficulty_map(df):
    """Calculate question difficulty as 1 - accuracy from training data."""
    q_dict = {}
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Computing difficulty"):
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


def load_pykt_data(data_dir, file_name, fold=0):
    """Load and parse pyKT-format CSV data."""
    file_path = os.path.join(data_dir, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    print(f"Loading CSV: {file_path}")
    df = pd.read_csv(file_path)

    print("Parsing string sequences...")
    for col in ['questions', 'concepts', 'responses', 'timestamps']:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: list(map(int, str(x).split(','))) if pd.notnull(x) else []
            )

    train_df = df[df['fold'] != fold].copy()
    valid_df = df[df['fold'] == fold].copy()

    print(f"Split - Train: {len(train_df)}, Valid (Fold {fold}): {len(valid_df)}")

    diff_map = get_difficulty_map(train_df)

    def df_to_list(dataframe):
        data_list = []
        for _, row in dataframe.iterrows():
            item = {
                'uid': row['uid'],
                'q_seq': row['questions'],
                'c_seq': row['concepts'],
                'r_seq': row['responses'],
                't_seq': row['timestamps'] if 'timestamps' in row else [0] * len(row['questions']),
            }
            data_list.append(item)
        return data_list

    return df_to_list(train_df), df_to_list(valid_df), diff_map


class PyKT_Dataset(Dataset):
    """PyTorch Dataset for knowledge tracing interaction sequences."""

    def __init__(self, data, diff_map, seq_len=200):
        self.data = data
        self.diff_map = diff_map
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        q_seq = row['q_seq']
        diff_seq = [self.diff_map.get(q, 50) for q in q_seq]
        type_seq = [1] * len(q_seq)

        q_tensor = torch.tensor(q_seq, dtype=torch.long)
        c_tensor = torch.tensor(row['c_seq'], dtype=torch.long)
        r_tensor = torch.tensor(row['r_seq'], dtype=torch.long)
        t_tensor = torch.tensor(row['t_seq'], dtype=torch.long)
        type_tensor = torch.tensor(type_seq, dtype=torch.long)
        diff_tensor = torch.tensor(diff_seq, dtype=torch.long)

        delta_t = torch.zeros_like(t_tensor, dtype=torch.float)
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
        }


def collate_fn(batch, seq_len=200):
    """Collate function with padding and truncation."""
    batch_data = {}
    r_seqs = [x['r_seq'] for x in batch]
    batch_data['r_seq'] = pad_sequence(r_seqs, batch_first=True, padding_value=-1)

    for key in ['q_seq', 'c_seq', 'type_seq', 'diff_seq']:
        seqs = [x[key] for x in batch]
        seqs = [s[-seq_len:] for s in seqs]
        batch_data[key] = pad_sequence(seqs, batch_first=True, padding_value=0)

    dt_seqs = [x['delta_t'] for x in batch]
    dt_seqs = [s[-seq_len:] for s in dt_seqs]
    batch_data['delta_t'] = pad_sequence(dt_seqs, batch_first=True, padding_value=0)

    if batch_data['r_seq'].size(1) > seq_len:
        batch_data['r_seq'] = batch_data['r_seq'][:, -seq_len:]

    return batch_data
