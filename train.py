import argparse
import json
import os
import random

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .data import DiscourseDataset, load_frame, stratified_split
from .model import RoleAwareInteractionModel


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def score(model, loader, device):
    model.eval(); labels, predictions = [], []
    for batch in loader:
        y = batch.pop("label").to(device)
        batch.pop("row_id")
        logits = model(**{k: v.to(device) for k, v in batch.items()})
        labels.extend(y.cpu().numpy().astype(int).tolist())
        predictions.extend((torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(int).tolist())
    p, r, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary", zero_division=0)
    return {"precision": float(p), "recall": float(r), "f1": float(f1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True); parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="bert-base-uncased")
    parser.add_argument("--max_length", type=int, default=128); parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10); parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01); parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=3); parser.add_argument("--projection_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variant", choices=["full", "no_role", "no_interaction", "no_adaptive", "no_role_interaction"], default="full")
    args = parser.parse_args(); os.makedirs(args.output_dir, exist_ok=True); set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = load_frame(args.data); train_df, val_df, test_df = stratified_split(frame, args.seed)
    for name, split in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        split.to_csv(os.path.join(args.output_dir, f"{name}_split.csv"), index=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    loaders = {}
    for name, split, shuffle in [("train", train_df, True), ("val", val_df, False), ("test", test_df, False)]:
        loaders[name] = DataLoader(DiscourseDataset(split, tokenizer, args.max_length), batch_size=args.batch_size, shuffle=shuffle)
    model = RoleAwareInteractionModel(args.model_name, args.projection_dim, args.dropout, args.variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(loaders["train"]) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)
    criterion = torch.nn.BCEWithLogitsLoss(); best_f1, stale = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in tqdm(loaders["train"], desc=f"epoch {epoch}"):
            labels = batch.pop("label").to(device); batch.pop("row_id")
            optimizer.zero_grad(set_to_none=True)
            logits = model(**{k: v.to(device) for k, v in batch.items()})
            loss = criterion(logits, labels); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
        metrics = score(model, loaders["val"], device); print({"epoch": epoch, **metrics})
        if metrics["f1"] > best_f1:
            best_f1, stale = metrics["f1"], 0
            torch.save({"state_dict": model.state_dict(), "args": vars(args)}, os.path.join(args.output_dir, "best_model.pt"))
        else:
            stale += 1
            if stale >= args.patience: break
    checkpoint = torch.load(os.path.join(args.output_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    result = {"validation_best_f1": best_f1, "test": score(model, loaders["test"], device)}
    with open(os.path.join(args.output_dir, "results.json"), "w", encoding="utf-8") as f: json.dump(result, f, indent=2)
    print(result)


if __name__ == "__main__": main()

