"""CLIP-style dual encoder with two independent pretrained backbones."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18
from transformers import AutoModel


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()  # ResNet18 feature size: 512

    def forward(self, images):
        return self.backbone(images)  # [B, 3, 224, 224] -> [B, 512]


class TextEncoder(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.output_dim = self.backbone.config.hidden_size

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state  # [B, L, hidden_size]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)  # [B, L, 1]
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, embed_dim=256):
        super().__init__()
        self.linear = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        return self.linear(x)


class CLIPModel(nn.Module):
    def __init__(self, embed_dim=256, text_model="distilbert-base-uncased"):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder(text_model)
        self.image_projection = ProjectionHead(512, embed_dim)
        self.text_projection = ProjectionHead(self.text_encoder.output_dim, embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode_image(self, images):
        features = self.image_encoder(images)
        return F.normalize(self.image_projection(features), p=2, dim=-1)

    def encode_text(self, input_ids, attention_mask):
        features = self.text_encoder(input_ids, attention_mask)
        return F.normalize(self.text_projection(features), p=2, dim=-1)

    def forward(self, images, input_ids, attention_mask):
        image_embeddings = self.encode_image(images)  # [B, D]
        text_embeddings = self.encode_text(input_ids, attention_mask)  # [B, D]
        scale = self.logit_scale.exp().clamp(max=100)
        return scale * (image_embeddings @ text_embeddings.T)  # [B, B]


def clip_loss(logits):
    """Symmetric in-batch InfoNCE; assumes diagonal is the only positive."""
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError("Expected a square [B, B] logits matrix.")
    if logits.shape[0] < 2:
        raise ValueError("Contrastive training needs at least 2 pairs per batch.")
    labels = torch.arange(logits.shape[0], device=logits.device)
    return (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.T, labels)
    ) / 2
