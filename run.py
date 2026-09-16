#!/usr/bin/env python3
"""HyConvKT - Unified entry point for training, evaluation, and ablation."""

import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader

from data import load_pykt_data, PyKT_Dataset, collate_fn, process_xes3g5m
from models import HyConvKT
from engines.trainer import Trainer
from engines.evaluator import Evaluator
from losses import HyConvKTLoss
from utils import set_seed


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_model(cfg, num_questions, num_concepts, device):
    return HyConvKT(
        num_questions=num_questions,
        num_concepts=num_concepts,
        embedding_dim=cfg['model']['embedding_dim'],
        d1=cfg['model']['d1'],
        d2=cfg['model']['d2'],
        num_difficulties=cfg['model']['num_difficulties'],
        num_types=cfg['model']['num_types'],
        hyperedge_nodes=cfg['model']['hyperedge_nodes'],
        kernel_size=tuple(cfg['model']['kernel_size']),
        conv_channels=cfg['model']['conv_channels'],
        dropout=cfg['model']['dropout'],
        device=device,
    )


def run_train(cfg, device):
    """Train HyConvKT model."""
    set_seed(cfg['seed'])

    train_data, valid_data, diff_map = load_pykt_data(
        cfg['data']['data_dir'], cfg['data']['file_name'], cfg['data']['fold']
    )

    max_q = max(max(row['q_seq']) for row in train_data + valid_data if row['q_seq']) + 1
    max_c = max(max(row['c_seq']) for row in train_data + valid_data if row['c_seq']) + 1
    print(f"Questions: {max_q}, Concepts: {max_c}")

    seq_len = cfg['data']['seq_len']
    train_dataset = PyKT_Dataset(train_data, diff_map, seq_len)
    valid_dataset = PyKT_Dataset(valid_data, diff_map, seq_len)

    collate = lambda batch: collate_fn(batch, seq_len=seq_len)
    train_loader = DataLoader(train_dataset, batch_size=cfg['training']['batch_size'],
                              shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg['training']['batch_size'],
                              shuffle=False, collate_fn=collate)

    model = build_model(cfg, max_q, max_c, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'],
                                 weight_decay=cfg['training']['weight_decay'])
    criterion = HyConvKTLoss()

    evaluator = Evaluator(model, criterion, device)
    trainer = Trainer(model, optimizer, criterion, cfg, device)

    best_auc = trainer.fit(train_loader, valid_loader, max_q, max_c,
                           cfg['model']['num_difficulties'], cfg['model']['num_types'],
                           evaluator)
    print(f"\nTraining complete. Best AUC: {best_auc:.4f}")


def run_test(cfg, device, checkpoint_path=None):
    """Evaluate HyConvKT model on validation set."""
    set_seed(cfg['seed'])

    train_data, valid_data, diff_map = load_pykt_data(
        cfg['data']['data_dir'], cfg['data']['file_name'], cfg['data']['fold']
    )

    max_q = max(max(row['q_seq']) for row in train_data + valid_data if row['q_seq']) + 1
    max_c = max(max(row['c_seq']) for row in train_data + valid_data if row['c_seq']) + 1

    seq_len = cfg['data']['seq_len']
    valid_dataset = PyKT_Dataset(valid_data, diff_map, seq_len)
    collate = lambda batch: collate_fn(batch, seq_len=seq_len)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg['training']['batch_size'],
                              shuffle=False, collate_fn=collate)

    model = build_model(cfg, max_q, max_c, device).to(device)
    criterion = HyConvKTLoss()
    evaluator = Evaluator(model, criterion, device)

    ckpt = checkpoint_path or os.path.join(cfg['checkpoint']['save_dir'], cfg['checkpoint']['model_name'])
    evaluator.load_checkpoint(ckpt)

    val_loss, val_auc, val_acc = evaluator.evaluate(
        valid_loader, max_q, max_c, cfg['model']['num_difficulties'], cfg['model']['num_types']
    )
    print(f"Test Results - Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, ACC: {val_acc:.4f}")


def run_ablation(cfg, device):
    """Run ablation study by disabling each module."""
    set_seed(cfg['seed'])

    print("=" * 60)
    print("Ablation Study: HyConvKT")
    print("=" * 60)

    train_data, valid_data, diff_map = load_pykt_data(
        cfg['data']['data_dir'], cfg['data']['file_name'], cfg['data']['fold']
    )

    max_q = max(max(row['q_seq']) for row in train_data + valid_data if row['q_seq']) + 1
    max_c = max(max(row['c_seq']) for row in train_data + valid_data if row['c_seq']) + 1

    seq_len = cfg['data']['seq_len']
    valid_dataset = PyKT_Dataset(valid_data, diff_map, seq_len)
    collate = lambda batch: collate_fn(batch, seq_len=seq_len)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg['training']['batch_size'],
                              shuffle=False, collate_fn=collate)

    # Full model
    print("\n[1/4] Full HyConvKT")
    model = build_model(cfg, max_q, max_c, device).to(device)
    ckpt = os.path.join(cfg['checkpoint']['save_dir'], cfg['checkpoint']['model_name'])
    if os.path.exists(ckpt):
        checkpoint = torch.load(ckpt, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        criterion = HyConvKTLoss()
        evaluator = Evaluator(model, criterion, device)
        _, auc, acc = evaluator.evaluate(
            valid_loader, max_q, max_c, cfg['model']['num_difficulties'], cfg['model']['num_types']
        )
        print(f"  Full Model     -> AUC: {auc:.4f}, ACC: {acc:.4f}")
    else:
        print("  [!] No checkpoint found. Train the model first.")
        return

    print("\n[Note] Ablation variants require retraining with modified architecture.")
    print("       See README.md for manual ablation instructions.")


def main():
    parser = argparse.ArgumentParser(description="HyConvKT - Hypergraph Convolutional Knowledge Tracing")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "test", "ablation", "preprocess"],
                        help="Mode: train, test, ablation, or preprocess")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to YAML configuration file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint for testing")

    args = parser.parse_args()
    cfg = load_config(args.config)

    if torch.cuda.is_available() and "cuda" in cfg.get("device", ""):
        device = torch.device(cfg["device"])
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    if args.mode == "preprocess":
        raw_path = os.path.join(cfg['data']['data_dir'], "..", "XES3G5M", "train_valid_sequences.csv")
        output_dir = cfg['data']['data_dir']
        process_xes3g5m(raw_path, output_dir,
                        min_seq_len=cfg['data']['min_seq_len'],
                        max_seq_len=cfg['data']['seq_len'])
    elif args.mode == "train":
        run_train(cfg, device)
    elif args.mode == "test":
        run_test(cfg, device, args.checkpoint)
    elif args.mode == "ablation":
        run_ablation(cfg, device)


if __name__ == "__main__":
    main()
