"""Parameter-free semantic consensus and structure-guided propagation."""
import torch
from torch import nn
import torch.nn.functional as F


class BSCM(nn.Module):
    """Batch-aware Semantic Consensus Module; one prototype per image."""
    def __init__(self, temperature=0.05):
        super().__init__()
        if temperature <= 0:
            raise ValueError('Semantic temperature must be positive')
        self.temperature = temperature

    def forward(self, features):
        descriptors = features.mean(dim=(-2, -1))
        normalized = F.normalize(descriptors, dim=1)
        weights = (normalized @ normalized.t() / self.temperature).softmax(dim=-1)
        prototypes = F.normalize(weights @ descriptors, dim=1)
        similarity = (F.normalize(features, dim=1) * prototypes[:, :, None, None]).sum(1, keepdim=True)
        return torch.sigmoid(similarity / self.temperature)


class SAPM(nn.Module):
    """Structure-guided Affinity Propagation Module using DINO cosine affinity."""
    def __init__(self, temperature=0.05):
        super().__init__()
        if temperature <= 0:
            raise ValueError('Structural temperature must be positive')
        self.temperature = temperature

    def forward(self, semantic_mask, dino_features):
        batch, channels, height, width = dino_features.shape
        normalized = F.normalize(dino_features, dim=1).reshape(batch, channels, -1)
        affinity = (normalized.transpose(1, 2).bmm(normalized) / self.temperature).softmax(-1)
        return affinity.bmm(semantic_mask.reshape(batch, -1, 1)).reshape(batch, 1, height, width)


class CMCA(nn.Module):
    """BSCM → SAPM → shared residual feature modulation."""
    def __init__(self, semantic_temperature=0.05, structural_temperature=0.05):
        super().__init__()
        self.bscm = BSCM(semantic_temperature)
        self.sapm = SAPM(structural_temperature)
        self.capture_vis = False
        self.vis_maps = {}

    def forward(self, clip_features, dino_features):
        if clip_features.ndim != 4 or clip_features.shape != dino_features.shape:
            raise ValueError('CMCA expects aligned CLIP/DINO tensors with identical BCHW shapes')
        semantic_mask = self.bscm(clip_features)
        refined_mask = self.sapm(semantic_mask, dino_features)
        self.vis_maps = ({'bscm': semantic_mask.detach(), 'sapm': refined_mask.detach()}
                         if self.capture_vis else {})
        return clip_features * (1 + refined_mask), dino_features * (1 + refined_mask)
