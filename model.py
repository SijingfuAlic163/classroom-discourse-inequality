import torch
from torch import nn
from transformers import AutoModel


def masked_mean(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


class RoleAwareInteractionModel(nn.Module):
    component_names = ("teacher", "student", "difference", "product")

    def __init__(self, model_name="bert-base-uncased", projection_dim=256,
                 dropout=0.2, variant="full"):
        super().__init__()
        self.variant = variant
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.teacher_projection = nn.Sequential(
            nn.Linear(hidden, projection_dim), nn.GELU(), nn.LayerNorm(projection_dim)
        )
        self.student_projection = nn.Sequential(
            nn.Linear(hidden, projection_dim), nn.GELU(), nn.LayerNorm(projection_dim)
        )
        self.shared_projection = nn.Sequential(
            nn.Linear(hidden, projection_dim), nn.GELU(), nn.LayerNorm(projection_dim)
        )
        self.component_transform = nn.ModuleList([
            nn.Sequential(nn.Linear(projection_dim, projection_dim), nn.GELU())
            for _ in self.component_names
        ])
        self.gate = nn.Sequential(
            nn.Linear(projection_dim * 4, projection_dim), nn.Tanh(),
            nn.Linear(projection_dim, 4)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(projection_dim, projection_dim // 2),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(projection_dim // 2, 1)
        )

    def encode(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return masked_mean(output.last_hidden_state, attention_mask)

    def forward(self, teacher_input_ids, teacher_attention_mask,
                student_input_ids, student_attention_mask, return_details=False):
        teacher_raw = self.encode(teacher_input_ids, teacher_attention_mask)
        student_raw = self.encode(student_input_ids, student_attention_mask)

        role_aware = self.variant not in {"no_role", "no_role_interaction"}
        interaction = self.variant not in {"no_interaction", "no_role_interaction"}
        adaptive = self.variant != "no_adaptive"

        if role_aware:
            teacher = self.teacher_projection(teacher_raw)
            student = self.student_projection(student_raw)
        else:
            teacher = self.shared_projection(teacher_raw)
            student = self.shared_projection(student_raw)

        if interaction:
            difference = torch.abs(teacher - student)
            product = teacher * student
        else:
            difference = torch.zeros_like(teacher)
            product = torch.zeros_like(teacher)

        raw_components = [teacher, student, difference, product]
        components = torch.stack([
            transform(value) for transform, value in zip(self.component_transform, raw_components)
        ], dim=1)
        concatenated = torch.cat(raw_components, dim=-1)
        if adaptive:
            weights = torch.softmax(self.gate(concatenated), dim=-1)
        else:
            weights = torch.full(
                (teacher.shape[0], 4), 0.25, device=teacher.device, dtype=teacher.dtype
            )
        fused = (components * weights.unsqueeze(-1)).sum(dim=1)
        logits = self.classifier(fused).squeeze(-1)
        if return_details:
            return {"logits": logits, "weights": weights, "components": components, "fused": fused}
        return logits

