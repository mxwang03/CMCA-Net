"""Shared CMCA-Net implementation for PASCAL VOC and MS COCO."""
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
import clip
from clip.clip_text import (new_class_names, new_class_names_coco,
                            BACKGROUND_CATEGORY, BACKGROUND_CATEGORY_COCO)
from clip.clip_tool import (generate_clip_fts, generate_cam_label,
                            perform_single_voc_cam, perform_single_coco_cam)
from .consensus import CMCA
from .segformer_head import SegFormerHead
from .Decoder.TransDecoder import DecoderTransformer
from .PAR import PAR


def reshape_transform(tensor, height, width):
    tensor = tensor.permute(1, 0, 2)
    return tensor[:, 1:].reshape(tensor.shape[0], height, width, tensor.shape[-1]).permute(0, 3, 1, 2)


@torch.no_grad()
def zeroshot_classifier(names, model):
    device = next(model.parameters()).device
    weights = []
    for name in names:
        tokens = clip.tokenize(['a clean origami {}.'.format(name)]).to(device)
        embedding = F.normalize(model.encode_text(tokens), dim=-1).mean(0)
        weights.append(F.normalize(embedding, dim=0))
    return torch.stack(weights)


class CMCANet(nn.Module):
    def __init__(self, num_classes=None, clip_model=None, dino_model='dinov2_vits14',
                 dino_fts_dim=384, decoder_layers=3, embedding_dim=256,
                 in_channels=(768, 768, 768, 768), dataset_root_path=None,
                 clip_flag=16, device='cuda', dataset='voc',
                 dino_repo_path='pretrained/dinov2', dino_checkpoint=None,
                 cmca_start_iter=10000, semantic_temperature=0.05,
                 structural_temperature=0.05):
        super().__init__()
        if dataset not in ('voc', 'coco'):
            raise ValueError('dataset must be voc or coco')
        expected_classes = 21 if dataset == 'voc' else 81
        if num_classes is not None and num_classes != expected_classes:
            raise ValueError('{} requires {} output classes'.format(dataset, expected_classes))
        if cmca_start_iter < 0:
            raise ValueError('cmca_start_iter must be nonnegative')
        dino_repo = Path(dino_repo_path).expanduser().resolve()
        if not (dino_repo / 'hubconf.py').is_file():
            raise FileNotFoundError('Set dino_init.repo_path to a local DINOv2 repository: {}'.format(dino_repo))
        if not dino_checkpoint or not Path(dino_checkpoint).is_file():
            raise FileNotFoundError('Set dino_init.checkpoint to local DINOv2 pretrained weights')
        self.dataset = dataset
        self.num_classes = expected_classes
        self.embedding_dim = embedding_dim
        self.clip_flag = clip_flag
        self.cmca_start_iter = cmca_start_iter
        self.iter_num = 0
        self.encoder, _ = clip.load(clip_model, device=device, jit=False)
        self.encoder.requires_grad_(False).eval()
        self.dino_encoder = torch.hub.load(str(dino_repo), dino_model, source='local', pretrained=False)
        weights = torch.load(dino_checkpoint, map_location='cpu', weights_only=True)
        self.dino_encoder.load_state_dict(weights, strict=True)
        self.dino_encoder.to(device).requires_grad_(False).eval()
        self.decoder_fts_fuse = SegFormerHead(in_channels=in_channels, embedding_dim=embedding_dim,
                                              num_classes=self.num_classes, index=1)
        self.dino_decoder_fts_fuse = SegFormerHead(in_channels=[dino_fts_dim] * 4,
                                                   embedding_dim=embedding_dim,
                                                   num_classes=self.num_classes, index=1)
        self.decoder = DecoderTransformer(width=embedding_dim, layers=decoder_layers,
                                           heads=8, output_dim=self.num_classes)
        self.cmca = CMCA(semantic_temperature, structural_temperature)
        self.register_buffer('bg_text_features', None, persistent=False)
        self.register_buffer('fg_text_features', None, persistent=False)
        self.root_path = str(Path(dataset_root_path or '.') / 'JPEGImages')
        self.cam_bg_thres = 1
        self.par = PAR(num_iter=20, dilations=[1, 2, 4, 8, 12, 24])
        self.grad_cam = None
        self.to(device)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        self.dino_encoder.eval()
        return self

    def get_param_groups(self):
        parameters = (list(self.decoder.parameters()) + list(self.decoder_fts_fuse.parameters())
                      + list(self.dino_decoder_fts_fuse.parameters()))
        return [[], [], [], parameters]

    def _prepare_cams(self):
        if self.grad_cam is None:
            from pytorch_grad_cam import GradCAM
            self.bg_text_features = zeroshot_classifier(
                BACKGROUND_CATEGORY if self.dataset == 'voc' else BACKGROUND_CATEGORY_COCO, self.encoder)
            self.fg_text_features = zeroshot_classifier(
                new_class_names if self.dataset == 'voc' else new_class_names_coco, self.encoder)
            self.grad_cam = GradCAM(model=self.encoder,
                                    target_layers=[self.encoder.visual.transformer.resblocks[-1].ln_1],
                                    reshape_transform=reshape_transform, clip_flag=self.clip_flag)

    def forward(self, img, img_names=None, mode='train', vis_only=False,
                cls_labels=None, iteration=None):
        batch, _, height, width = img.shape
        if min(height, width) < max(14, self.clip_flag):
            raise ValueError('Input image is smaller than a backbone patch')
        compute_pseudo = self.training and mode == 'train' and not vis_only
        if compute_pseudo:
            if cls_labels is None or tuple(cls_labels.shape) != (batch, self.num_classes - 1):
                raise ValueError('Training requires image-level cls_labels with shape [B, foreground_classes]')
            self._prepare_cams()
            self.iter_num = int(iteration) if iteration is not None else self.iter_num + 1
        original_size = (height, width)
        patch_size = self.clip_flag
        aligned_size = (max(patch_size, round(height / patch_size) * patch_size),
                        max(patch_size, round(width / patch_size) * patch_size))
        if original_size != aligned_size:
            img = F.interpolate(img, size=aligned_size, mode='bilinear', align_corners=False)
        height, width = img.shape[-2:]
        with torch.no_grad():
            clip_tokens, clip_attention = generate_clip_fts(img, self.encoder, require_all_fts=True,
                                                            clip_flag=self.clip_flag)
            dino_h, dino_w = (height // 14) * 14, (width // 14) * 14
            dino_input = F.interpolate(img, size=(dino_h, dino_w), mode='bilinear', align_corners=False)
            dino_tokens = self.dino_encoder.forward_features(dino_input)['x_norm_patchtokens']
        stack = torch.stack(clip_tokens)
        clip_dense = stack[-1, 1:].permute(1, 2, 0).reshape(batch, -1, height // patch_size, width // patch_size)
        clip_features = self.decoder_fts_fuse(clip_dense.unsqueeze(0))
        dino_dense = dino_tokens.reshape(batch, dino_h // 14, dino_w // 14, -1).permute(0, 3, 1, 2)
        dino_features = self.dino_decoder_fts_fuse(dino_dense.unsqueeze(0))
        dino_features = F.interpolate(dino_features, size=clip_features.shape[-2:], mode='bilinear', align_corners=False)
        if not self.training or self.iter_num > self.cmca_start_iter:
            enhanced_clip, enhanced_dino = self.cmca(clip_features, dino_features)
        else:
            enhanced_clip, enhanced_dino = clip_features, dino_features
            self.cmca.vis_maps = {}
        clip_logits, _ = self.decoder(enhanced_clip)
        dino_logits, _ = self.decoder(enhanced_dino)
        if vis_only:
            return clip_logits, dino_logits
        if not compute_pseudo:
            return clip_logits, dino_logits, None, None
        fused_probabilities = (0.5 * (clip_logits + dino_logits)).softmax(dim=1).detach()
        dense = torch.cat([clip_features, dino_features], dim=1).flatten(2)
        affinity = torch.sigmoid(dense.transpose(1, 2).bmm(dense))
        cam_features = stack[-2].unsqueeze(0).permute(2, 1, 0, 3)
        cam_attention = torch.stack(clip_attention).permute(1, 0, 2, 3)
        generate = perform_single_voc_cam if self.dataset == 'voc' else perform_single_coco_cam
        pseudo_labels = []
        for index in range(batch):
            class_ids = torch.nonzero(cls_labels[index], as_tuple=False).flatten().tolist()
            if not class_ids:
                pseudo_labels.append(torch.zeros(original_size, device=img.device, dtype=torch.long))
                continue
            with torch.enable_grad():
                refined, keys, cam_w, cam_h = generate(
                    '', img[index], cam_features[index], cam_attention[index], affinity[index:index + 1],
                    self.bg_text_features, self.fg_text_features, self.grad_cam,
                    mode='train', require_seg_trans=True, seg_dino_cam=fused_probabilities[index],
                    clip_flag=self.clip_flag, class_ids=class_ids)
            cams = generate_cam_label(refined, keys, cam_w, cam_h)['refined_cam'].to(img.device)
            background = (1 - cams.max(dim=0, keepdim=True).values).pow(self.cam_bg_thres)
            cams = torch.cat([background, cams], dim=0)
            valid_keys = torch.tensor([0] + [int(key) + 1 for key in keys], device=img.device)
            with torch.no_grad():
                pseudo = valid_keys[self.par(img[index:index + 1].float(), cams[None].float()).argmax(1)]
                if aligned_size != original_size:
                    pseudo = F.interpolate(pseudo[:, None].float(), size=original_size, mode='nearest')[:, 0].long()
                pseudo_labels.append(pseudo[0])
        return clip_logits, dino_logits, torch.stack(pseudo_labels), affinity
