import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer

class MindScopeDataset(Dataset):
    def __init__(self, csv_file, tokenizer_name="distilbert-base-uncased", max_len=64):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

        self.tab_cols = ["age", "sleep_hours", "stress_level", "family_history"]
        self.text_col = "clinical_note"
        self.target_col = "risk_label"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # 1) Text (NLP)
        text = str(row[self.text_col])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # 2) Tabular (Normalized)
        # Age: /100, Sleep: /24, Stress: /10, Family history: already 0/1
        tabular_features = torch.tensor([
            float(row["age"]) / 100.0,
            float(row["sleep_hours"]) / 24.0,
            float(row["stress_level"]) / 10.0,
            float(row["family_history"])
        ], dtype=torch.float32)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "tabular_features": tabular_features,
            "label": torch.tensor(int(row[self.target_col]), dtype=torch.long)
        }
