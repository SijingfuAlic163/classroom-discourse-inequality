import re
from dataclasses import dataclass

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


REQUIRED_COLUMNS = {"teacher_text", "student_text", "label"}


def normalize_text(text):
    text = re.sub(r"[\x00-\x1f\x7f]", " ", str(text))
    return re.sub(r"\s+", " ", text).strip()


def load_frame(path):
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame = frame.dropna(subset=list(REQUIRED_COLUMNS)).copy()
    frame["teacher_text"] = frame["teacher_text"].map(normalize_text)
    frame["student_text"] = frame["student_text"].map(normalize_text)
    frame["label"] = frame["label"].astype(int)
    if not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError("label must contain only 0 and 1")
    return frame.reset_index(drop=True)


def stratified_split(frame, seed=42):
    train_val, test = train_test_split(
        frame, test_size=0.20, random_state=seed, stratify=frame["label"]
    )
    train, val = train_test_split(
        train_val, test_size=0.10, random_state=seed,
        stratify=train_val["label"]
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


class DiscourseDataset(Dataset):
    def __init__(self, frame, tokenizer, max_length=128):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        teacher = self.tokenizer(
            row.teacher_text, max_length=self.max_length, truncation=True,
            padding="max_length", return_tensors="pt"
        )
        student = self.tokenizer(
            row.student_text, max_length=self.max_length, truncation=True,
            padding="max_length", return_tensors="pt"
        )
        item = {
            "teacher_input_ids": teacher["input_ids"].squeeze(0),
            "teacher_attention_mask": teacher["attention_mask"].squeeze(0),
            "student_input_ids": student["input_ids"].squeeze(0),
            "student_attention_mask": student["attention_mask"].squeeze(0),
            "label": torch.tensor(float(row.label)),
            "row_id": torch.tensor(index),
        }
        return item

