"""Helpers for visualizing CMCA responses without annotation-dependent inference."""
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from .runtime import model_kwargs, load_checkpoint
from .inference import image_tensor


def create_model(cfg, checkpoint):
    from .model_attn_aff_voc import CMCANet
    model = CMCANet(**model_kwargs(cfg))
    load_checkpoint(model, checkpoint)
    return model.eval()


def load_image(image_id, root, resize_long, device):
    with Image.open(Path(root) / (image_id + '.jpg')) as opened:
        image = opened.convert('RGB')
    inputs = image_tensor(image, device)
    height, width = inputs.shape[-2:]
    ratio = resize_long / max(height, width)
    inputs = F.interpolate(inputs, size=(max(16, round(height * ratio)), max(16, round(width * ratio))), mode='bilinear', align_corners=False)
    return np.array(image), torch.cat([inputs, inputs.flip(-1)])
