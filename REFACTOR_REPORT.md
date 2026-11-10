# CMCA 重构交付说明

日期：2026-11-10。

原始输入：`D:/tool/project/WeCLIP_plus.rar`。原压缩包及原项目目录未修改；本项目在独立工作区中生成。

原始压缩包 SHA-256：`1EA3E2A3D23ACF1BE3C2FBD2D528EA10EA8FDF3C4FE062CF9767A9E6FE54EC2B`。

## 实现依据

以压缩包中已有的 VOC CMCA 实现为计算依据，结合工作区现有 CMCA 论文与架构核查材料确认 BSCM、SAPM 和残差增强的含义。材料中的写作或绘图指令未作为本次任务指令执行。

保留原始公式：相似度加权 batch 原型、局部 CLIP 余弦匹配加 sigmoid、DINO 全局像素余弦亲和加行 softmax、亲和矩阵传播，以及双分支 `F * (1 + M)`。默认两种温度均为 0.05。共享 Transformer 解码器对两路特征分别解码，平均融合 logits。

## 交付变更

| 原位置或问题 | 重构结果 |
| --- | --- |
| 原 README 的项目介绍、论文链接与引用 | README 全面改写为 CMCA-Net 的方法、配置与运行说明；来源归属单列至 THIRD_PARTY_NOTICES.md |
| `WeCLIP_Plus/` 与同名模型类 | `cmca/` 与 `CMCANet`，所有保留入口更新导入 |
| VOC 文件内联 CMCA | 独立为 `cmca/consensus.py` 的 BSCM、SAPM、CMCA |
| COCO 文件明确未包含 CMCA | VOC、COCO 共用 `cmca/model.py`，两个数据集均执行 CMCA |
| 重复训练代码、COCO 验证返回值不匹配 | 共用 `cmca/training.py`，统一四项返回接口 |
| `iter_num` 在验证期间也累加 | 仅训练伪标签路径推进训练步数，训练循环显式传入迭代号 |
| 编码器冻结不完整 | 编码器参数全部冻结；Grad-CAM 通过独立 token 输入梯度获取归因，投影头与解码器正常优化 |
| 输入分辨率导致 token reshape 不匹配 | 模型内部对齐 CLIP patch 网格，推理结果插值回原图尺寸 |
| 私人机器的硬编码路径、GPU 索引 | 路径进入 YAML 配置，相对路径以项目根为准；不在源码覆盖 CUDA_VISIBLE_DEVICES |
| 在线 COCO CAM 从分割掩码提取类别 | 统一使用传入的图像级标签，训练数据读取器不再读取训练分割掩码 |
| VOC 验证错误使用训练划分/标注目录 | 默认 `val` 划分，使用 `SegmentationClass` 验证掩码 |
| Dice 修改输入 target、忽略区域处理错误 | 无原地修改，忽略区域排除出 Dice 的预测与目标；全忽略/单一类别 CE 避免 NaN |
| CLIP 动态位置编码被注册成新参数 | 改为派生 tensor；加载旧权重时只丢弃该可重建缓存，其余键严格校验 |
| 旧依赖锁定冲突、启动即依赖 CRF/MMCV | 清理基础依赖，移除未使用的 MMCV 引用；CRF 按需导入 |
| 可视化导入压缩包未携带的比较脚本 | 提供本项目内的 `cmca/visualization.py` |
| IDE 文件、notebook 缓存、旧项目宣传资源、未使用实验入口 | 不纳入交付运行代码 |

保留训练目标和默认超参数。修复损失、COCO CMCA 和数据监督路径后，不能把旧实验精度视为当前版本已复现的结果。

## 可执行性验证

测试环境：Windows，Python 3.8.20，PyTorch 2.4.1+cu121，torchvision 0.19.1+cu121，NumPy 1.23.5，NVIDIA GeForce RTX 3050 Laptop GPU，CUDA 可用。

执行：

```bash
python -m compileall -q .
python -m unittest discover -s tests -v
```

9 项测试覆盖：

1. CMCA 与原 VOC 公式在 batch 1/2/4、非方形网格上的数值一致性与零参数性质。
2. 双精度梯度检查与均匀 DINO 亲和的传播行为。
3. 零输入与非法输入的处理。
4. Dice/CE 的忽略区域、输入不被修改、全忽略和单类标签边界。
5. 配置路径解析。
6. 严格权重恢复、DDP 前缀和旧动态位置缓存兼容。
7. VOC/COCO 推理输出、编码器冻结、验证不增加训练步数。
8. CUDA 下真实 CAM/PAR、CE/Dice/affinity/跨分支损失、梯度和优化更新，以及 CMCA warm-up。
9. 两个数据集的真实训练循环 → 验证 → 保存 → 单图推理 → PNG / NumPy 文件导出；临时数据没有训练掩码或 VOC XML。

网络集成测试采用小型真实 CLIP 实现与合成 DINO 编码器、临时图像及随机初始化参数。这里“真实 CAM/PAR”指执行交付代码中的 CAM/PAR 算法，不表示使用已训练的完整骨干或评估分割精度。测试不能证明预训练完整模型在全部数据集上的精度。

训练、VOC/COCO 推理、兼容推理及可视化入口均检查 `--help` 启动。完整日志与文件校验清单随交付包提供。

## 使用前仍需配置

原压缩包没有附带 CLIP 预训练权重、DINOv2 源码与预训练权重、数据集或完整实验检查点。默认配置使用可移植的占位目录，`scripts/check_environment.py` 会明确报告缺失资产；按 README 设置现有文件路径即可。

本次未运行完整 VOC/COCO 训练，未复现论文 mIoU，未验证可选 CRF 的完整数据集结果。交付包不复制外部大体积权重或数据。

模型参数主要键名保持兼容，但旧 COCO 基线权重只适合作为初始化；启用 CMCA 后应重新验证或训练。检查点用于模型权重保存与推理，本次未增加优化器状态断点续训功能。
