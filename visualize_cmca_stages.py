# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from omegaconf import OmegaConf

# Reuse the checkpoint loading and preprocessing already used for CAMs.
from cmca.visualization import create_model, load_image
from cmca.runtime import load_config
from utils.imutils import encode_cmap


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/voc_attn_reg.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--ids", nargs="+", required=True)
    p.add_argument("--output-dir", default="results/cmca_stages")
    p.add_argument("--gt-dir", default=None)
    p.add_argument("--vmin", type=float, default=0.0)
    p.add_argument("--vmax", type=float, default=1.0)
    return p.parse_args()


def merge_pair(tensor, output_size):
    """Unflip the second view, average, and resize to original image size."""
    merged = 0.5 * (
        tensor[0:1].float()
        + tensor[1:2].float().flip(-1)
    )
    return F.interpolate(
        merged,
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )


def mask_rgb(mask):
    mask = np.asarray(mask, dtype=np.uint8)
    valid_mask = mask.copy()
    valid_mask[mask == 255] = 0
    color = np.asarray(encode_cmap(valid_mask), dtype=np.uint8).copy()
    color[mask == 255] = (180, 180, 180)
    return color


def heat_overlay(image, response, cmap, norm, alpha=0.45):
    # One shared scale for every BSCM/SAPM response.
    # No independent per-image or per-map normalization.
    color = np.asarray(cmap(norm(response))[:, :, :3])
    mixed = (
        (1.0 - alpha) * image.astype(np.float32) / 255.0
        + alpha * color
    )
    return np.round(np.clip(mixed, 0, 1) * 255).astype(np.uint8)


def panel_canvas(array, size=(320, 240)):
    """Equal-sized panels, preserving aspect ratio and complete content."""
    image = Image.fromarray(np.asarray(array, dtype=np.uint8))
    return ImageOps.pad(
        image,
        size,
        method=Image.Resampling.LANCZOS
        if hasattr(Image, "Resampling")
        else Image.LANCZOS,
        color="white",
        centering=(0.5, 0.5),
    )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required by the current model.")
    if args.vmin >= args.vmax:
        raise ValueError("--vmin must be smaller than --vmax.")

    torch.manual_seed(1)
    np.random.seed(1)

    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    root = Path(str(cfg.dataset.root_dir))
    gt_dir = (
        Path(args.gt_dir)
        if args.gt_dir
        else root / "SegmentationClass"
    )
    image_ids = list(dict.fromkeys(Path(x).stem for x in args.ids))

    # Check paths before initializing the large model.
    for image_id in image_ids:
        for path in (
            root / "JPEGImages" / (image_id + ".jpg"),
            gt_dir / (image_id + ".png")):
            if not path.is_file():
                raise FileNotFoundError(str(path))

    model = create_model(cfg, args.checkpoint)
    model.cmca.capture_vis = True
    model.eval()

    cmap = plt.get_cmap("jet")
    norm = Normalize(vmin=args.vmin, vmax=args.vmax, clip=True)
    rows = []

    for image_id in image_ids:
        image, inputs = load_image(
            image_id,
            model.root_path,
            cfg.clip_init.resize_long,
            "cuda",
        )
        height, width = image.shape[:2]
        model.cmca.vis_maps = {}

        # Safe because vis_only skips the GradCAM path.
        with torch.no_grad():
            seg_clip, seg_dino = model(
                inputs,
                [image_id, image_id],
                mode="val",
                vis_only=True,
            )

            maps = model.cmca.vis_maps
            if not {"bscm", "sapm"}.issubset(maps):
                raise RuntimeError(
                    "No responses captured. Check the model modifications."
                )

            ms = merge_pair(
                maps["bscm"], (height, width)
            )[0, 0].cpu().numpy()

            mr = merge_pair(
                maps["sapm"], (height, width)
            )[0, 0].cpu().numpy()

            logits = 0.5 * (seg_clip.float() + seg_dino.float())
            logits = merge_pair(logits, (height, width))
            prediction = logits.argmax(dim=1)[0].cpu().numpy()
            prediction = prediction.astype(np.uint8)

        gt = np.asarray(
            Image.open(gt_dir / (image_id + ".png")),
            dtype=np.uint8,
        )
        if gt.shape != (height, width):
            raise ValueError("GT/image size mismatch: " + image_id)

        example_dir = output_dir / image_id
        example_dir.mkdir(parents=True, exist_ok=True)

        # Preserve numerical responses, including their native resolution.
        np.savez_compressed(
            str(example_dir / "responses.npz"),
            ms=ms,
            mr=mr,
            ms_native=maps["bscm"].float().cpu().numpy(),
            mr_native=maps["sapm"].float().cpu().numpy(),
            prediction=prediction,
        )

        panels = [
            image,
            mask_rgb(gt),
            heat_overlay(image, ms, cmap, norm),
            heat_overlay(image, mr, cmap, norm),
            mask_rgb(prediction),
        ]
        names = ["input", "gt", "bscm", "sapm", "prediction"]

        for name, panel in zip(names, panels):
            Image.fromarray(panel).save(
                str(example_dir / (name + ".png"))
            )

        rows.append(panels)
        print(
            "{} | Ms: {:.4f}~{:.4f}, std={:.5f} | "
            "Mr: {:.4f}~{:.4f}, std={:.5f}".format(
                image_id,
                float(ms.min()), float(ms.max()), float(ms.std()),
                float(mr.min()), float(mr.max()), float(mr.std()),
            )
        )

    # Five equal-sized columns; original images are never stretched.
    titles = [
        "Input", "Ground Truth",
        r"BSCM $M^S$", r"SAPM $M^R$", "Prediction",
    ]
    fig, axes = plt.subplots(
        len(rows), 5,
        figsize=(13.0, 2.05 * len(rows) + 0.7),
        squeeze=False,
    )

    for r, panels in enumerate(rows):
        for c, panel in enumerate(panels):
            axes[r, c].imshow(panel_canvas(panel))
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(
                    titles[c], fontsize=12, fontfamily="serif"
                )

    fig.subplots_adjust(
        left=0.01, right=0.99,
        bottom=0.11, top=0.94,
        wspace=0.025, hspace=0.035,
    )
    colorbar_ax = fig.add_axes([0.38, 0.035, 0.24, 0.018])
    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        cax=colorbar_ax,
        orientation="horizontal",
    )
    colorbar.set_label("Response value", fontsize=10)

    fig.savefig(
        str(output_dir / "cmca_stages.png"),
        dpi=300, bbox_inches="tight", pad_inches=0.02,
    )
    fig.savefig(
        str(output_dir / "cmca_stages.pdf"),
        dpi=300, bbox_inches="tight", pad_inches=0.02,
    )
    plt.close(fig)
    print("Saved to:", output_dir.resolve())


if __name__ == "__main__":
    main()
