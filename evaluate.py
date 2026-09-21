import argparse
import json
import os

import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .data import DiscourseDataset, load_frame
from .model import RoleAwareInteractionModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True); parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True); parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args(); os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device); cfg = checkpoint["args"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    frame = load_frame(args.data)
    loader = DataLoader(DiscourseDataset(frame, tokenizer, cfg["max_length"]), batch_size=args.batch_size)
    model = RoleAwareInteractionModel(cfg["model_name"], cfg["projection_dim"], cfg["dropout"], cfg["variant"]).to(device)
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    records, attrs = [], []
    for batch in loader:
        labels = batch.pop("label"); row_ids = batch.pop("row_id")
        inputs = {k: v.to(device) for k, v in batch.items()}
        with torch.enable_grad():
            details = model(**inputs, return_details=True)
            components = details["components"]; components.retain_grad()
            probability = torch.sigmoid(details["logits"])
            model.zero_grad(set_to_none=True); probability.sum().backward()
            gradient_scores = (components.grad * components).sum(-1).abs()
            gradient_scores = gradient_scores / gradient_scores.sum(-1, keepdim=True).clamp(min=1e-9)
        for i, row_id in enumerate(row_ids.tolist()):
            prob = float(probability[i].detach().cpu())
            records.append({"row_id": row_id, "label": int(labels[i]), "probability": prob, "prediction": int(prob >= 0.5)})
            entry = {"row_id": row_id}
            for j, name in enumerate(model.component_names):
                entry[f"adaptive_weight_{name}"] = float(details["weights"][i, j].detach().cpu())
                entry[f"gradient_attribution_{name}"] = float(gradient_scores[i, j].detach().cpu())
            attrs.append(entry)
    pred = pd.DataFrame(records).sort_values("row_id")
    pred.to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)
    pd.DataFrame(attrs).sort_values("row_id").to_csv(os.path.join(args.output_dir, "attributions.csv"), index=False)
    p, r, f1, _ = precision_recall_fscore_support(pred.label, pred.prediction, average="binary", zero_division=0)
    metrics = {"precision": float(p), "recall": float(r), "f1": float(f1)}
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f: json.dump(metrics, f, indent=2)
    pd.DataFrame(confusion_matrix(pred.label, pred.prediction), index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(os.path.join(args.output_dir, "confusion_matrix.csv"))
    print(metrics)


if __name__ == "__main__": main()

