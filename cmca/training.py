import argparse
import datetime
import logging
import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from cmca.runtime import load_config, model_kwargs
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datasets import voc, coco
from utils.losses import get_aff_loss, get_seg_loss
from utils import evaluate
from utils.AverageMeter import AverageMeter
from utils.camutils import cams_to_affinity_label
from utils.optimizer import PolyWarmupAdamW
from cmca.model_attn_aff_voc import CMCANet as VOCNet
from cmca.model_attn_aff_coco import CMCANet as COCONet
from cmca.dice_loss import DiceLoss


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_logger(filename='test.log'):
    ## setup logger
    logFormatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fHandler = logging.FileHandler(filename, mode='w')
    fHandler.setFormatter(logFormatter)
    logger.addHandler(fHandler)

    cHandler = logging.StreamHandler()
    cHandler.setFormatter(logFormatter)
    logger.addHandler(cHandler)


def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)

    scale = (total_iter-cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta*scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)


@torch.no_grad()
def validate(model=None, data_loader=None, cfg=None):
    model.eval()
    hist = np.zeros((cfg.dataset.num_classes, cfg.dataset.num_classes))
    score = None
    for name, inputs, labels, cls_label in tqdm(data_loader, ncols=100):
        clip_logits, dino_logits, _, _ = model(inputs.cuda(), name, mode='val')
        logits = F.interpolate(0.5 * (clip_logits + dino_logits), size=labels.shape[1:], mode='bilinear', align_corners=False)
        hist, score = evaluate.scores(labels.numpy(), logits.argmax(1).cpu().numpy(), hist, num_classes=cfg.dataset.num_classes)
    model.train()
    return score, None


def get_mask_by_radius(h=20, w=20, radius=8):
    hw = h * w
    mask  = np.zeros((hw, hw))
    for i in range(hw):
        _h = i // w
        _w = i % w

        _h0 = max(0, _h - radius)
        _h1 = min(h, _h + radius+1)
        _w0 = max(0, _w - radius)
        _w1 = min(w, _w + radius+1)
        for i1 in range(_h0, _h1):
            for i2 in range(_w0, _w1):
                _i2 = i1 * w + i2
                mask[i, _i2] = 1
                mask[_i2, i] = 1

    return mask



def train(cfg, dataset='voc', radius=8):
    if not torch.cuda.is_available():
        raise RuntimeError('CMCA-Net training requires a CUDA-enabled PyTorch environment')
    if cfg.dataset.crop_size % cfg.clip_init.clip_flag:
        raise ValueError('Training crop_size must be a multiple of clip_flag')
    dataset_module = voc if dataset == 'voc' else coco
    classification_dataset = dataset_module.VOC12ClsDataset if dataset == 'voc' else dataset_module.CocoClsDataset
    segmentation_dataset = dataset_module.VOC12SegDataset if dataset == 'voc' else dataset_module.CocoSegDataset
    model_class = VOCNet if dataset == 'voc' else COCONet


    num_workers = cfg.runtime.num_workers
    
    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)
    
    train_dataset = classification_dataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.train.split,
        stage='train',
        aug=True,
        resize_range=cfg.dataset.resize_range,
        rescale_range=cfg.dataset.rescale_range,
        crop_size=cfg.dataset.crop_size,
        img_fliplr=True,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )
    
    val_dataset = segmentation_dataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.val.split,
        stage='val',
        aug=False,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.train.samples_per_gpu,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=False,
                              drop_last=True,
                              **({"prefetch_factor": 4} if num_workers else {}))

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=False,
                            drop_last=False)


    model = model_class(**model_kwargs(cfg, device='cuda'))
    logging.info('\nNetwork config: \n%s'%(model))
    param_groups = model.get_param_groups()
    model.cuda()

    mask_size = int(cfg.dataset.crop_size // cfg.clip_init.clip_flag)
    attn_mask = get_mask_by_radius(h=mask_size, w=mask_size, radius=radius)
    writer = SummaryWriter(cfg.work_dir.tb_logger_dir)

    optimizer = PolyWarmupAdamW(
        params=[
            {
                "params": param_groups[0],
                "lr": cfg.optimizer.learning_rate,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": param_groups[1],
                "lr": 0.0,
                "weight_decay": 0.0,
            },
            {
                "params": param_groups[2],
                "lr": cfg.optimizer.learning_rate*10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": param_groups[3],
                "lr": cfg.optimizer.learning_rate*10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
        ],
        lr = cfg.optimizer.learning_rate,
        weight_decay = cfg.optimizer.weight_decay,
        betas = cfg.optimizer.betas,
        warmup_iter = cfg.scheduler.warmup_iter,
        max_iter = cfg.train.max_iters,
        warmup_ratio = cfg.scheduler.warmup_ratio,
        power = cfg.scheduler.power
    )
    logging.info('\nOptimizer: \n%s' % optimizer)

    if len(train_loader) == 0:
        raise ValueError('Training split is smaller than samples_per_gpu with drop_last=True')
    train_loader_iter = iter(train_loader)

    avg_meter = AverageMeter()

    criterion_dice = DiceLoss().cuda()


    for n_iter in range(cfg.train.max_iters):

        try:
            img_name, inputs, cls_labels, img_box = next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(train_loader)
            img_name, inputs, cls_labels, img_box = next(train_loader_iter)

        segs_clip, segs_dino, cam, attn_pred = model(inputs.cuda(), img_name, cls_labels=cls_labels, iteration=n_iter + 1)

        pseudo_label = cam

        segs= 0.5*segs_clip+0.5*segs_dino
        segs = F.interpolate(segs, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)

        segs_clip = F.interpolate(segs_clip, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)
        segs_dino = F.interpolate(segs_dino, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)

        pred_clip_max, pred_label_clip = torch.max(F.softmax(segs_clip, dim=1), dim=1)
        pred_dino_max, pred_label_dino = torch.max(F.softmax(segs_dino, dim=1), dim=1)
        pred_max, pred_label_seg = torch.max(F.softmax(segs, dim=1), dim=1)

        fts_cam = cam.clone()

        aff_label = cams_to_affinity_label(fts_cam, mask=attn_mask, ignore_index=cfg.dataset.ignore_index, clip_flag=cfg.clip_init.clip_flag)
        attn_loss, pos_count, neg_count = get_aff_loss(attn_pred, aff_label)

        if n_iter > 18000:
            pseudo_label[pred_max>0.75] = pred_label_seg[pred_max>0.75]
        seg_loss = get_seg_loss(segs, pseudo_label.type(torch.long), ignore_index=cfg.dataset.ignore_index)
        seg_loss2 = criterion_dice(segs, pseudo_label.type(torch.long))

        seg_clip_loss2 = get_seg_loss(segs_clip, pred_label_dino.type(torch.long), ignore_index=cfg.dataset.ignore_index)
        seg_dino_loss2 = get_seg_loss(segs_dino, pred_label_clip.type(torch.long), ignore_index=cfg.dataset.ignore_index)


        loss = 1 * seg_loss + 0.1 * attn_loss + 0.1 * (seg_clip_loss2 + seg_dino_loss2) + 1*seg_loss2

        avg_meter.add({'seg_loss': seg_loss.item(), 'attn_loss': attn_loss.item()})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (n_iter + 1) % cfg.train.log_iters == 0:

            delta, eta = cal_eta(time0, n_iter+1, cfg.train.max_iters)
            cur_lr = optimizer.param_groups[0]['lr']

            preds = torch.argmax(segs,dim=1).cpu().numpy().astype(np.int16)
            gts = pseudo_label.cpu().numpy().astype(np.int16)

            seg_mAcc = (preds==gts).sum()/preds.size

            logging.info("Iter: %d; Elasped: %s; ETA: %s; LR: %.3e;, pseudo_seg_loss: %.4f, attn_loss: %.4f, pseudo_seg_mAcc: %.4f"%(n_iter+1, delta, eta, cur_lr, avg_meter.pop('seg_loss'), avg_meter.pop('attn_loss'),  seg_mAcc))

            writer.add_scalars('train/loss',  {"seg_loss": seg_loss.item(), "attn_loss": attn_loss.item()}, global_step=n_iter)


        if (n_iter +1) % cfg.train.eval_iters == 0:
            ckpt_name = os.path.join(cfg.work_dir.ckpt_dir, "cmca_iter_%d.pth"%(n_iter+1))
            logging.info('Validating...')
            torch.save(model.state_dict(), ckpt_name)
            seg_score, _ = validate(model=model, data_loader=val_loader, cfg=cfg)
            logging.info("segs score:")
            logging.info(seg_score)

    if cfg.train.max_iters % cfg.train.eval_iters:
        torch.save(model.state_dict(), os.path.join(cfg.work_dir.ckpt_dir, 'cmca_iter_{}.pth'.format(cfg.train.max_iters)))
    writer.close()
    return True


def main(default_dataset='voc'):
    parser = argparse.ArgumentParser(description='Train CMCA-Net')
    parser.add_argument('--config', default='configs/{}_attn_reg.yaml'.format(default_dataset))
    parser.add_argument('--work_dir', default=None)
    parser.add_argument('--radius', default=8, type=int)
    parser.add_argument('--seed', default=1, type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.work_dir is not None:
        cfg.work_dir.dir = args.work_dir

    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())

    cfg.work_dir.ckpt_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.ckpt_dir, timestamp)
    cfg.work_dir.pred_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.pred_dir)
    cfg.work_dir.tb_logger_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.tb_logger_dir, timestamp)

    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.tb_logger_dir, exist_ok=True)

    setup_logger(filename=os.path.join(cfg.work_dir.dir, timestamp+'.log'))
    logging.info('\nargs: %s' % args)
    logging.info('\nconfigs: %s' % cfg)

    setup_seed(args.seed)
    train(cfg=cfg, dataset=default_dataset, radius=args.radius)
