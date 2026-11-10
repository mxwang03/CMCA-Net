# CMCA-Net

CMCA-Net 面向图像级标签监督的弱监督语义分割。核心流程为：冻结的 CLIP / DINOv2 编码器 → 特征投影 → BSCM → SAPM → 双分支残差增强 → 共享 Transformer 解码器 → 分支 logits 平均融合。

## CMCA 实现

- **BSCM**（Batch-aware Semantic Consensus Module）：以每张图像为查询，对 mini-batch 的 CLIP 全局描述子进行相似度加权，形成图像专属原型，再生成单通道语义响应。
- **SAPM**（Structure-guided Affinity Propagation Module）：通过 DINO 特征的全局像素亲和矩阵传播语义响应。
- **残差增强**：`F_clip' = F_clip * (1 + M_refined)`，`F_dino' = F_dino * (1 + M_refined)`。
- BSCM / SAPM 不引入可学习参数，语义与结构温度默认均为 `0.05`。
- VOC、COCO 共用同一模型和训练流程；输出类别数分别为 21、81（含背景）。
- 训练前 10,000 步旁路 CMCA，第 10,001 步开始启用；验证与推理始终启用。验证不会增加训练步数。
- 推理时原图与水平翻转图组成两视图 batch，分别在各尺度中构造语义共识。

## 环境

建议 Python 3.10。先安装与 GPU 匹配的 PyTorch / torchvision，再安装其余依赖：

```bash
python -m pip install -r requirements.txt
```

本次代码测试环境为 Python 3.8.20、PyTorch 2.4.1+cu121、torchvision 0.19.1+cu121、NumPy 1.23.5。训练和在线 Grad-CAM 需要 CUDA；分割推理支持 `--device cpu`。基础流程不需要 MMCV。只有使用 `--crf` 时才需要额外安装 `requirements-crf.txt`。

## 准备编码器

配置文件不包含私人机器的绝对路径。相对路径统一相对于项目根目录解析。

```text
pretrained/
├── ViT-B-16.pt
├── dinov2/                    # 官方 DINOv2 源码，含 hubconf.py
└── dinov2_vits14_pretrain.pth  # 与 dinov2_vits14 对应的权重
```

准备 [DINOv2 官方源码](https://github.com/facebookresearch/dinov2)，并将对应编码器权重放入上方目录，或修改配置中的路径。DINOv2 的本地加载接口已对照[官方实现](https://github.com/facebookresearch/dinov2/blob/main/dinov2/hub/backbones.py)核对。CLIP 使用本项目携带的 dense-token 实现，无需另装同名 `clip` 包。

模型构造采用本地 DINOv2 源码与显式权重加载，不会自动下载 DINOv2。更换编码器时，需同步修改 `dino_model`、`dino_fts_fuse_dim` 和 `checkpoint`。

## 数据与配置

修改 `configs/voc_attn_reg.yaml` 或 `configs/coco_attn_reg.yaml` 中的：

| 配置项 | 内容 |
| --- | --- |
| `dataset.root_dir` | 数据集根目录 |
| `dataset.name_list_dir` | 划分列表和图像级类别标签目录 |
| `clip_init.clip_pretrain_path` | CLIP 权重 |
| `dino_init.repo_path` | 本地 DINOv2 源码目录 |
| `dino_init.checkpoint` | 对应 DINOv2 权重 |
| `cmca.start_iter` | CMCA 延迟启用步数 |
| `runtime.num_workers` | 数据加载进程数，调试可设为 0 |

VOC 目录：

```text
VOC2012/
├── JPEGImages/
└── SegmentationClass/        # 仅验证/评估使用
```

COCO 目录（与数据读取器一致）：

```text
MSCOCO/
├── JPEGImages/
│   ├── train/train2014/COCO_train2014_000000000001.jpg
│   └── val/val2014/COCO_val2014_000000000001.jpg
└── SegmentationClass/
    ├── train/000000000001.png
    └── val/000000000001.png
```

训练只读取 RGB 图像、划分列表和 `cls_labels_onehot.npy` 图像级标签，不读取训练分割掩码或 VOC XML。COCO 的训练掩码目录可省略。类别标签向量不含背景，长度分别为 20 和 80；COCO 掩码须使用连续类别 ID 0–80，255 为忽略区域。VOC 的 `train_aug.txt` 保留增强训练划分，训练图像需覆盖该列表。

先检查依赖和路径：

```bash
python scripts/check_environment.py --config configs/voc_attn_reg.yaml
python scripts/check_environment.py --config configs/coco_attn_reg.yaml
```

## 训练

```bash
python scripts/dist_clip_voc.py --config configs/voc_attn_reg.yaml
python scripts/dist_clip_coco.py --config configs/coco_attn_reg.yaml
```

这是单 GPU 训练入口。用环境变量 `CUDA_VISIBLE_DEVICES` 选择 GPU。默认保留 VOC 30,000 步 / batch 4、COCO 80,000 步 / batch 8；显存不足可在配置中调整。

训练目标为 `L_CE + L_Dice + 0.1 L_aff + 0.1 (L_clip←dino + L_dino←clip)`。训练日志、TensorBoard 和检查点位于配置的 `work_dir`。检查点命名为 `cmca_iter_<step>.pth`，在验证间隔及最后一步保存。

## 推理

```bash
python test_msc_flip_voc.py --config configs/voc_attn_reg.yaml --model_path path/to/cmca_iter_30000.pth
python test_msc_flip_coco.py --config configs/coco_attn_reg.yaml --model_path path/to/cmca_iter_80000.pth
```

单张图像无需标签或分割标注：

```bash
python test_msc_flip_voc.py --model_path path/to/cmca_iter_30000.pth --input path/to/image.jpg --scales 1.0 --work_dir results/demo
```

默认启用尺度 `[1.0, 1.5]` 与水平翻转，平均融合 logits 后得到预测。可使用 `--device cpu` 或可选 `--crf`。`test_msc_flip_seg.py` 为 VOC 推理兼容入口。

输出至 `<work_dir>/<eval_set>/prediction/`（类别 ID PNG）、`prediction_cmap/`（彩色 PNG）及 `logit/`（每图 CHW logits）。只有找到验证掩码时才计算并打印相应图像的 mIoU；不会把无标注图片计入评估。

模型前向默认返回 `(clip_logits, dino_logits, pseudo_labels, affinity)`。纯推理时后两项为 `None`，`vis_only=True` 时仅返回两路 logits。训练调用须额外传入 `cls_labels`。

## 模块可视化

```bash
python visualize_cmca_stages.py --config configs/voc_attn_reg.yaml --checkpoint path/to/cmca_iter_30000.pth --ids 2007_000032 --output-dir results/cmca_stages
```

该 VOC 可视化工具输出 BSCM / SAPM 响应及预测对照，需要相应图像与 GT；GT 仅用于展示。

## 验证与迁移

```bash
python -m compileall -q .
python -m unittest discover -s tests -v
```

测试覆盖公式一致性、梯度、warm-up、忽略标签、严格权重加载，以及 VOC/COCO 的训练→验证→保存→单图推理流程。网络集成测试使用小型真实 CLIP 和合成 DINO 编码器，不依赖外部数据和预训练权重；CUDA 不可用时跳过训练集成测试。

保留编码器、投影头和解码器的主要 state-dict 键名。加载器支持纯 `state_dict`、`state_dict` / `model` 包装及 `module.` 前缀；仅丢弃可重新生成的旧 `encoder.visual.positional_embedding_new` 缓存，其余键严格校验。旧 COCO 基线权重可用于结构初始化，但不等于经过 CMCA 训练的权重。

当前压缩包未附带预训练编码器、DINOv2 源码、数据集或完整实验检查点。测试通过说明软件流程可运行，不能替代真实编码器和完整数据集上的训练与精度复现。详细变更、来源信息和验证边界见 `REFACTOR_REPORT.md` 与 `THIRD_PARTY_NOTICES.md`。

## 代码结构

```text
cmca/consensus.py       BSCM / SAPM / CMCA
cmca/model.py           共享 CMCA-Net
cmca/training.py        共享训练与验证
cmca/inference.py       多尺度推理与导出
cmca/runtime.py         配置和检查点加载
scripts/               VOC/COCO 训练入口与环境检查
configs/               数据集、编码器、CMCA 和训练配置
datasets/              数据读取器、划分和图像级标签
clip/                  CLIP dense-token 实现
pytorch_grad_cam/      在线 CAM 计算
tests/                 回归和集成测试
```
