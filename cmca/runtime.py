"""Configuration and strict checkpoint loading shared by all entry points."""
from pathlib import Path
import os
from collections import OrderedDict
from omegaconf import OmegaConf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        config_path = PROJECT_ROOT / path
    cfg = OmegaConf.load(config_path)
    for section, key in [('dataset', 'root_dir'), ('dataset', 'name_list_dir'),
                         ('clip_init', 'clip_pretrain_path'), ('dino_init', 'repo_path'),
                         ('dino_init', 'checkpoint'), ('work_dir', 'dir')]:
        value = Path(os.path.expandvars(str(cfg[section][key]))).expanduser()
        cfg[section][key] = str(value if value.is_absolute() else PROJECT_ROOT / value)
    return cfg


def model_kwargs(cfg, device='cuda'):
    return dict(num_classes=cfg.dataset.num_classes, clip_model=cfg.clip_init.clip_pretrain_path,
                dino_model=cfg.dino_init.dino_model, dino_fts_dim=cfg.dino_init.dino_fts_fuse_dim,
                decoder_layers=cfg.dino_init.decoder_layer, embedding_dim=cfg.clip_init.embedding_dim,
                in_channels=cfg.clip_init.in_channels, dataset_root_path=cfg.dataset.root_dir,
                clip_flag=cfg.clip_init.clip_flag, device=device,
                dino_repo_path=cfg.dino_init.repo_path, dino_checkpoint=cfg.dino_init.checkpoint,
                cmca_start_iter=cfg.cmca.start_iter,
                semantic_temperature=cfg.cmca.semantic_temperature,
                structural_temperature=cfg.cmca.structural_temperature)


def load_checkpoint(model, path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    state = payload.get('state_dict', payload.get('model', payload))
    state = OrderedDict((key[7:] if key.startswith('module.') else key, value)
                        for key, value in state.items())
    # Older checkpoints registered a dynamically resized position encoding.
    # It is reconstructed from positional_embedding at each input resolution.
    state.pop('encoder.visual.positional_embedding_new', None)
    return model.load_state_dict(state, strict=True)
