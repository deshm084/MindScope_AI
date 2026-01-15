import torch
import torch.nn as nn
from transformers import DistilBertModel

class MindScopeFusionModel(nn.Module):
    def __init__(self, num_tabular_features, num_classes=3, freeze_bert=True):
        super(MindScopeFusionModel, self).__init__()

        # --- Tower 1: NLP ---
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.bert_drop = nn.Dropout(0.3)
        self.bert_out_dim = 768

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        # --- Tower 2: Tabular (with BatchNorm) ---
        self.tab_net = nn.Sequential(
            nn.Linear(num_tabular_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.tab_out_dim = 32

        # --- Fusion Layer ---
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.bert_out_dim + self.tab_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_ids, attention_mask, tabular_features):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_text_output = bert_output.last_hidden_state[:, 0, :]
        pooled_text_output = self.bert_drop(pooled_text_output)

        tab_output = self.tab_net(tabular_features)

        combined = torch.cat((pooled_text_output, tab_output), dim=1)
        logits = self.fusion_layer(combined)
        return logits
