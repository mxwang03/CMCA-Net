"""Segmentation inference with paired views and optional CRF postprocessing."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .runtime import load_config, model_kwargs, load_checkpoint


@torch.no_grad()
def predict(model, inputs, resize_long=512, scales=(1.0, 1.5)):
    height, width = inputs.shape[-2:]
    outputs = []
    for scale in scales:
        ratio = resize_long * scale / max(height, width)
        size = (max(16, round(height * ratio)), max(16, round(width * ratio)))
        image = F.interpolate(inputs, size=size, mode='bilinear', align_corners=False)
        paired = torch.cat([image, image.flip(-1)])
        clip_logits, dino_logits, _, _ = model(paired, mode='val')
        logits = 0.5 * (clip_logits.float() + dino_logits.float())
        fused = (logits[0:1] + logits[1:2].flip(-1)) * 0.5
        outputs.append(F.interpolate(fused, size=(height, width), mode='bilinear', align_corners=False))
    return torch.stack(outputs).mean(0)


def image_tensor(image, device):
    # Match the supplied training pipeline's ImageNet normalization.
    array = np.asarray(image.convert('RGB'), dtype=np.float32)
    array = (array - np.array([123.675, 116.28, 103.53], dtype=np.float32)) / np.array([58.395, 57.12, 57.375], dtype=np.float32)
    return torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)


def main(default_dataset='voc'):
    parser = argparse.ArgumentParser(description='CMCA-Net semantic segmentation inference')
    parser.add_argument('--config', default='configs/{}_attn_reg.yaml'.format(default_dataset))
    parser.add_argument('--model_path', required=True, help='CMCA-Net state_dict checkpoint')
    parser.add_argument('--input', help='Optional single RGB image; no annotations required')
    parser.add_argument('--work_dir', default='results/cmca_{}'.format(default_dataset))
    parser.add_argument('--eval_set', default='val')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--scales', nargs='+', type=float, default=[1.0, 1.5])
    parser.add_argument('--crf', action='store_true', help='Enable optional DenseCRF')
    args = parser.parse_args()
    if any(scale <= 0 for scale in args.scales):
        parser.error('--scales must be positive')
    cfg = load_config(args.config)
    if default_dataset == 'voc':
        from .model_attn_aff_voc import CMCANet
    else:
        from .model_attn_aff_coco import CMCANet
    model = CMCANet(**model_kwargs(cfg, args.device))
    load_checkpoint(model, args.model_path)
    model.eval()
    root = Path(cfg.dataset.root_dir)
    if args.input:
        items = [(Path(args.input).stem, Path(args.input), None)]
    else:
        ids = np.atleast_1d(np.loadtxt(Path(cfg.dataset.name_list_dir) / (args.eval_set + '.txt'), dtype=str))
        if default_dataset == 'voc':
            items = [(str(key), root / 'JPEGImages' / (str(key) + '.jpg'),
                      root / 'SegmentationClass' / (str(key) + '.png')) for key in ids]
        else:
            split = 'train' if 'train' in args.eval_set else 'val'
            items = [(str(key).split('_')[-1], root / 'JPEGImages' / split / (split + '2014') / (str(key) + '.jpg'),
                      root / 'SegmentationClass' / split / (str(key).split('_')[-1] + '.png')) for key in ids]
    output = Path(args.work_dir) / args.eval_set
    for folder in ['prediction', 'prediction_cmap', 'logit']:
        (output / folder).mkdir(parents=True, exist_ok=True)
    from utils.imutils import encode_cmap
    from utils.evaluate import scores
    hist = np.zeros((cfg.dataset.num_classes, cfg.dataset.num_classes))
    evaluated = 0
    for key, image_path, label_path in tqdm(items):
        with Image.open(image_path) as opened:
            image = opened.convert('RGB')
        logits = predict(model, image_tensor(image, args.device), cfg.clip_init.resize_long, args.scales)
        probabilities = logits.softmax(1)[0].cpu().numpy()
        if args.crf:
            from utils.dcrf import crf_inference
            probabilities = crf_inference(np.array(image), probabilities, labels=cfg.dataset.num_classes)
        prediction = probabilities.argmax(0).astype(np.uint8)
        Image.fromarray(prediction).save(output / 'prediction' / (key + '.png'))
        Image.fromarray(encode_cmap(prediction)).save(output / 'prediction_cmap' / (key + '.png'))
        np.save(output / 'logit' / (key + '.npy'), logits[0].cpu().numpy())
        if label_path is not None and label_path.is_file():
            with Image.open(label_path) as opened:
                label = np.array(opened)
            hist, score = scores([label], [prediction], hist, num_classes=cfg.dataset.num_classes)
            evaluated += 1
    print('Saved predictions to {}'.format(output.resolve()))
    if evaluated:
        print('mIoU on {} images with ground truth: {:.4f}%'.format(evaluated, 100 * score['miou']))
