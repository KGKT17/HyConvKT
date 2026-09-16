import os
import pandas as pd
import numpy as np
from tqdm import tqdm


def process_xes3g5m(raw_path, output_dir, min_seq_len=3, max_seq_len=200, test_ratio=0.2):
    """Process raw XES3G5M CSV into pyKT-compatible format.

    Steps: parse strings, filter short sequences, remap IDs,
    split train/test by user, assign folds, window long sequences.
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data not found: {raw_path}. Please download it first.")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading raw data: {raw_path} ...")
    df = pd.read_csv(raw_path)
    print(f"  Raw rows: {len(df)}")

    print("Parsing string sequences...")
    for col in ['questions', 'concepts', 'responses', 'timestamps']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: list(map(int, str(x).split(','))) if pd.notnull(x) else [])

    df = df[df['questions'].apply(len) >= min_seq_len]
    print(f"  After filtering (len >= {min_seq_len}): {len(df)}")

    print("Remapping IDs to compact range...")
    all_q = set()
    all_c = set()
    for q_seq in df['questions']:
        all_q.update(q_seq)
    for c_seq in df['concepts']:
        all_c.update(c_seq)

    q2idx = {q: i for i, q in enumerate(sorted(all_q))}
    c2idx = {c: i for i, c in enumerate(sorted(all_c))}
    print(f"  Questions: {len(q2idx)}, Concepts: {len(c2idx)}")

    def remap(seq, mapping):
        return [mapping[x] for x in seq]

    df['questions'] = df['questions'].apply(lambda x: remap(x, q2idx))
    df['concepts'] = df['concepts'].apply(lambda x: remap(x, c2idx))

    print("Splitting train/test by user...")
    uids = df['uid'].unique()
    np.random.shuffle(uids)
    test_size = int(len(uids) * test_ratio)
    test_uids = set(uids[:test_size])

    train_uids = [u for u in uids if u not in test_uids]
    uid2fold = {uid: i % 5 for i, uid in enumerate(train_uids)}

    print("Windowing long sequences...")
    processed_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        uid = row['uid']
        q_seq = row['questions']
        c_seq = row['concepts']
        r_seq = row['responses']
        t_seq = row['timestamps']

        if uid in test_uids:
            fold = -1
            ftype = 'test'
        else:
            fold = uid2fold[uid]
            ftype = 'train'

        seq_len = len(q_seq)
        num_windows = (seq_len + max_seq_len - 1) // max_seq_len

        for w in range(num_windows):
            start = w * max_seq_len
            end = min((w + 1) * max_seq_len, seq_len)

            w_q = q_seq[start:end]
            w_c = c_seq[start:end]
            w_r = r_seq[start:end]
            w_t = t_seq[start:end]

            real_len = len(w_q)
            pad_len = max_seq_len - real_len
            select_masks = [1] * real_len + [-1] * pad_len

            final_q = w_q + [0] * pad_len
            final_c = w_c + [0] * pad_len
            final_r = w_r + [-1] * pad_len
            final_t = w_t + [0] * pad_len

            processed_rows.append({
                'fold': fold,
                'uid': uid,
                'questions': ",".join(map(str, final_q)),
                'concepts': ",".join(map(str, final_c)),
                'responses': ",".join(map(str, final_r)),
                'timestamps': ",".join(map(str, final_t)),
                'selectmasks': ",".join(map(str, select_masks)),
                'file_type': ftype,
            })

    final_df = pd.DataFrame(processed_rows)

    train_df = final_df[final_df['file_type'] == 'train'].drop(columns=['file_type'])
    train_df.to_csv(os.path.join(output_dir, "train_valid_sequences.csv"), index=False)

    test_df = final_df[final_df['file_type'] == 'test'].drop(columns=['file_type', 'fold'])
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print(f"\nDone! Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"Questions: {len(q2idx)}, Concepts: {len(c2idx)}")

    return q2idx, c2idx
