import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm

# ================= 配置 =================
# 1. 这里填写你下载的原始 XES3G5M 文件的路径
RAW_DATA_PATH = "./XES3G5M/train_valid_sequences.csv"

# 2. 这里填写输出 pyKT 格式数据的目录
OUTPUT_DIR = "./data/xes3g5m"

# 3. 参数配置
MIN_SEQ_LEN = 3
MAX_SEQ_LEN = 200
TEST_RATIO = 0.2  # 20% 用户作为纯测试集


# =======================================

def process_xes3g5m_real():
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"严重错误：找不到原始文件 {RAW_DATA_PATH}。请修改脚本中的 RAW_DATA_PATH 路径！")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"1. 正在读取原始数据: {RAW_DATA_PATH} ...")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"   原始数据行数: {len(df)}")

    # --- 1. 字符串解析 ---
    print("2. 解析序列格式...")
    # XES3G5M 原始格式是 csv 字符串，需解析
    for col in ['questions', 'concepts', 'responses', 'timestamps']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: list(map(int, str(x).split(','))))

    # --- 2. 过滤短序列 ---
    df = df[df['questions'].apply(len) >= MIN_SEQ_LEN]
    print(f"   过滤后行数 (Len >= {MIN_SEQ_LEN}): {len(df)}")

    # --- 3. ID 重映射 (Remap) ---
    # 这一步非常关键！pyKT 要求 ID 必须是 0 到 N-1 的紧凑整数
    print("3. 构建 ID 映射 (Remapping)...")

    all_q = set()
    all_c = set()
    for q_seq in df['questions']: all_q.update(q_seq)
    for c_seq in df['concepts']: all_c.update(c_seq)

    # 建立映射表: 原始ID -> 新ID (0 ~ N-1)
    q2idx = {q: i for i, q in enumerate(sorted(all_q))}
    c2idx = {c: i for i, c in enumerate(sorted(all_c))}

    print(f"   题目总数: {len(q2idx)}, 知识点总数: {len(c2idx)}")

    # 应用映射
    def remap(seq, mapping):
        return [mapping[x] for x in seq]

    df['questions'] = df['questions'].apply(lambda x: remap(x, q2idx))
    df['concepts'] = df['concepts'].apply(lambda x: remap(x, c2idx))

    # --- 4. 划分 Test Set (按用户划分) ---
    print("4. 划分训练/测试集...")
    uids = df['uid'].unique()
    np.random.shuffle(uids)
    test_size = int(len(uids) * TEST_RATIO)
    test_uids = set(uids[:test_size])

    # Fold 分配 (0-4)
    # 不在 test_uids 里的用户，分配 fold 0-4
    train_uids = [u for u in uids if u not in test_uids]
    uid2fold = {uid: i % 5 for i, uid in enumerate(train_uids)}

    # --- 5. 窗口切割与格式化 ---
    print("5. 执行窗口切割 (Windowing) 并生成最终数据...")
    processed_rows = []

    for _, row in tqdm(df.iterrows(), total = len(df)):
        uid = row['uid']
        q_seq = row['questions']
        c_seq = row['concepts']
        r_seq = row['responses']
        t_seq = row['timestamps']

        # 确定 fold 和 file_type
        if uid in test_uids:
            fold = -1
            ftype = 'test'
        else:
            fold = uid2fold[uid]
            ftype = 'train'

        seq_len = len(q_seq)

        # 切割长序列
        num_windows = (seq_len + MAX_SEQ_LEN - 1) // MAX_SEQ_LEN

        for w in range(num_windows):
            start = w * MAX_SEQ_LEN
            end = min((w + 1) * MAX_SEQ_LEN, seq_len)

            w_q = q_seq[start:end]
            w_c = c_seq[start:end]
            w_r = r_seq[start:end]
            w_t = t_seq[start:end]

            # Padding
            real_len = len(w_q)
            pad_len = MAX_SEQ_LEN - real_len

            # pyKT selectmasks: 1=有效, -1=Padding
            select_masks = [1] * real_len + [-1] * pad_len

            # 数据填充 (ID填0, Mask填-1)
            final_q = w_q + [0] * pad_len
            final_c = w_c + [0] * pad_len
            final_r = w_r + [-1] * pad_len  # Label Padding 设为 -1
            final_t = w_t + [0] * pad_len

            processed_rows.append({
                'fold': fold,
                'uid': uid,
                'questions': ",".join(map(str, final_q)),
                'concepts': ",".join(map(str, final_c)),
                'responses': ",".join(map(str, final_r)),
                'timestamps': ",".join(map(str, final_t)),
                'selectmasks': ",".join(map(str, select_masks)),
                'file_type': ftype
            })

    # --- 6. 保存 ---
    final_df = pd.DataFrame(processed_rows)

    # 保存 train_valid
    train_df = final_df[final_df['file_type'] == 'train'].drop(columns = ['file_type'])
    train_path = os.path.join(OUTPUT_DIR, "train_valid_sequences.csv")
    train_df.to_csv(train_path, index = False)

    # 保存 test
    test_df = final_df[final_df['file_type'] == 'test'].drop(columns = ['file_type', 'fold'])
    test_path = os.path.join(OUTPUT_DIR, "test.csv")
    test_df.to_csv(test_path, index = False)

    print("\n" + "=" * 30)
    print("处理完成！")
    print(f"训练验证集: {len(train_df)} 条序列 -> {train_path}")
    print(f"测试集:     {len(test_df)} 条序列 -> {test_path}")
    print(f"题目数量 (Compact ID): {len(q2idx)}")
    print(f"知识点数量 (Compact ID): {len(c2idx)}")
    print("=" * 30)


if __name__ == "__main__":
    process_xes3g5m_real()