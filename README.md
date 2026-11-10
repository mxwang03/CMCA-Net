# CMCA-Net

CMCA-Net is a framework for weakly supervised semantic segmentation using image-level labels. Its pipeline consists of frozen CLIP and DINOv2 encoders, feature projections, BSCM, SAPM, residual enhancement of both feature streams, a shared Transformer decoder, and average fusion of the branch logits.

## Method

- **BSCM (Batch-aware Semantic Consensus Module):** Uses each image as a query to aggregate global CLIP descriptors from the mini-batch with similarity-based weights. The resulting image-specific prototype is matched against local features to produce a single-channel semantic response.
- **SAPM (Structure-guided Affinity Propagation Module):** Propagates the semantic response through a global pixel affinity matrix derived from DINO features.
- **Residual enhancement:** Applies `F_clip' = F_clip * (1 + M_refined)` and `F_dino' = F_dino * (1 + M_refined)` before decoding.
- BSCM and SAPM introduce no learnable parameters. Both semantic and structural temperatures default to `0.05`.
- PASCAL VOC and MS COCO share the same model and training pipeline, with 21 and 81 output classes, respectively, including background.
- CMCA is bypassed during the first 10,000 training iterations and enabled from iteration 10,001 onward. It is always enabled during validation and inference. Validation does not advance the training iteration counter.
- During inference, the original image and its horizontally flipped view form a two-view batch for semantic consensus at each scale.

## Environment

Python 3.10 is recommended. Install a PyTorch / torchvision build suitable for your GPU, then install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

The code was tested with Python 3.8.20, PyTorch 2.4.1+cu121, torchvision 0.19.1+cu121, and NumPy 1.23.5. Training and online Grad-CAM generation require CUDA. Segmentation inference also supports `--device cpu`. The core pipeline does not require MMCV. Install the additional dependencies in `requirements-crf.txt` only if you use `--crf`.

## Encoder Preparation

Configuration files use portable paths. All relative paths are resolved from the project root.

```text
pretrained/
├── ViT-B-16.pt
├── dinov2/                    # Official DINOv2 source, including hubconf.py
└── dinov2_vits14_pretrain.pth  # Weights matching dinov2_vits14
```

Obtain the [official DINOv2 source](https://github.com/facebookresearch/dinov2) and place the corresponding encoder weights in the locations above, or update the configuration paths. The local DINOv2 loading interface was checked against the [official implementation](https://github.com/facebookresearch/dinov2/blob/main/dinov2/hub/backbones.py). CLIP uses the dense-token implementation bundled with this project; a separate package named `clip` is not required.

The model loads DINOv2 from local source and an explicit checkpoint. It does not download DINOv2 automatically. When changing the DINO encoder, update `dino_model`, `dino_fts_fuse_dim`, and `checkpoint` together.

## Data and Configuration

Update the following fields in `configs/voc_attn_reg.yaml` or `configs/coco_attn_reg.yaml`:

| Field | Description |
| --- | --- |
| `dataset.root_dir` | Dataset root directory |
| `dataset.name_list_dir` | Directory containing split lists and image-level class labels |
| `clip_init.clip_pretrain_path` | CLIP checkpoint path |
| `dino_init.repo_path` | Local DINOv2 source directory |
| `dino_init.checkpoint` | Matching DINOv2 checkpoint path |
| `cmca.start_iter` | Number of training iterations before CMCA is enabled |
| `runtime.num_workers` | Number of data-loading workers; set to 0 for debugging |

PASCAL VOC directory structure:

```text
VOC2012/
├── JPEGImages/
└── SegmentationClass/        # Used only for validation and evaluation
```

MS COCO directory structure, as expected by the data loader:

```text
MSCOCO/
├── JPEGImages/
│   ├── train/train2014/COCO_train2014_000000000001.jpg
│   └── val/val2014/COCO_val2014_000000000001.jpg
└── SegmentationClass/
    ├── train/000000000001.png
    └── val/000000000001.png
```

Training reads only RGB images, split lists, and the image-level labels in `cls_labels_onehot.npy`. It does not read training segmentation masks or VOC XML annotations. The COCO training mask directory can therefore be omitted. Class-label vectors exclude background and have lengths of 20 for VOC and 80 for COCO. COCO segmentation masks must use contiguous class IDs from 0 to 80, with 255 reserved for ignored pixels. The VOC `train_aug.txt` file contains the augmented training split; all images listed in it must be available.

Check dependencies and configured paths before training:

```bash
python scripts/check_environment.py --config configs/voc_attn_reg.yaml
python scripts/check_environment.py --config configs/coco_attn_reg.yaml
```

## Training

```bash
python scripts/dist_clip_voc.py --config configs/voc_attn_reg.yaml
python scripts/dist_clip_coco.py --config configs/coco_attn_reg.yaml
```

These are single-GPU training entry points. Select the GPU using the `CUDA_VISIBLE_DEVICES` environment variable. The default settings use 30,000 iterations with a batch size of 4 for VOC and 80,000 iterations with a batch size of 8 for COCO. Adjust the configuration if GPU memory is limited.

The training objective is `L_CE + L_Dice + 0.1 L_aff + 0.1 (L_clip←dino + L_dino←clip)`. Training logs, TensorBoard logs, and checkpoints are written under the configured `work_dir`. Checkpoints are named `cmca_iter_<step>.pth` and saved at validation intervals and at the final iteration.

## Inference

```bash
python test_msc_flip_voc.py --config configs/voc_attn_reg.yaml --model_path path/to/cmca_iter_30000.pth
python test_msc_flip_coco.py --config configs/coco_attn_reg.yaml --model_path path/to/cmca_iter_80000.pth
```

Single-image inference does not require class labels or segmentation annotations:

```bash
python test_msc_flip_voc.py --model_path path/to/cmca_iter_30000.pth --input path/to/image.jpg --scales 1.0 --work_dir results/demo
```

The default inference settings use scales `[1.0, 1.5]` and horizontal flipping. Predictions are obtained by averaging the logits across branches and views. Use `--device cpu` for CPU inference or the optional `--crf` flag for DenseCRF postprocessing. `test_msc_flip_seg.py` is a compatibility entry point for VOC inference.

Outputs are saved under `<work_dir>/<eval_set>/`:

- `prediction/`: PNG masks containing class IDs.
- `prediction_cmap/`: Colorized PNG masks.
- `logit/`: Per-image logits in CHW format.

When validation masks are available, mIoU is computed and printed for the corresponding images. Images without ground-truth masks are excluded from evaluation.

The model normally returns `(clip_logits, dino_logits, pseudo_labels, affinity)`. During segmentation inference, the last two values are `None`. With `vis_only=True`, only the two branch logits are returned. Training calls must also supply `cls_labels`.

## Module Visualization

```bash
python visualize_cmca_stages.py --config configs/voc_attn_reg.yaml --checkpoint path/to/cmca_iter_30000.pth --ids 2007_000032 --output-dir results/cmca_stages
```

This VOC visualization tool displays BSCM and SAPM responses alongside segmentation predictions. It requires the corresponding images and ground-truth masks; ground truth is used only for display.

## Validation and Checkpoint Migration

```bash
python -m compileall -q .
python -m unittest discover -s tests -v
```

The tests cover formula equivalence, gradients, delayed CMCA activation, ignored labels, strict checkpoint loading, and the VOC/COCO training → validation → checkpoint saving → single-image inference workflow. Network integration tests use a small instance of the real CLIP implementation and a synthetic DINO encoder. They do not require external datasets or pretrained weights. Training integration tests are skipped when CUDA is unavailable.

The main state-dict keys for the encoders, projection heads, and decoder are preserved. The checkpoint loader accepts plain state dictionaries, dictionaries wrapped under `state_dict` or `model`, and keys prefixed with `module.`. Only the legacy `encoder.visual.positional_embedding_new` cache is discarded because it can be regenerated; all other keys are checked strictly. Older COCO baseline checkpoints can initialize the model, but they are not checkpoints trained with CMCA enabled.

This repository does not include pretrained encoder weights, DINOv2 source, datasets, or full experimental checkpoints. Passing the tests verifies the software workflow; it does not replace training and accuracy evaluation with the full encoders and datasets.

## Project Structure

```text
cmca/consensus.py       BSCM / SAPM / CMCA
cmca/model.py           Shared CMCA-Net model
cmca/training.py        Shared training and validation
cmca/inference.py       Multi-scale inference and export
cmca/runtime.py         Configuration and checkpoint loading
scripts/               VOC/COCO training entry points and environment checks
configs/               Dataset, encoder, CMCA, and training settings
datasets/              Data loaders, split lists, and image-level labels
clip/                  CLIP dense-token implementation
pytorch_grad_cam/      Online CAM generation
tests/                 Regression and integration tests
```

## Acknowledgements

Our implementation builds upon [WeCLIP+](https://github.com/zbf1991/WeCLIP/tree/main/WeCLIP%2B), the baseline introduced in [Frozen CLIP-DINO: A Strong Backbone for Weakly Supervised Semantic Segmentation](https://ieeexplore.ieee.org/document/10891864) (TPAMI 2025). We thank the authors for making their code publicly available.

We also thank the authors of [AFA](https://github.com/rulixiang/afa) and [CLIP-ES](https://github.com/linyq2117/CLIP-ES) for the components inherited through the baseline, and the authors of [CLIP](https://github.com/openai/CLIP), [DINOv2](https://github.com/facebookresearch/dinov2), and [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) for their contributions to the community.
