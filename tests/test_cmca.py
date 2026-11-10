"""Formula, gradient, checkpoint and training/inference regression checks.

The network integration tests use a small real CLIP and a synthetic DINO
encoder, so they need no external weights. They do not measure accuracy.
"""
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from cmca.consensus import BSCM, SAPM, CMCA
from cmca.dice_loss import dice_loss
from cmca.runtime import load_checkpoint, load_config
from utils.losses import get_seg_loss, get_aff_loss


class ConsensusTests(unittest.TestCase):
    def test_matches_original_voc_formula(self):
        torch.manual_seed(42)
        for batch in (1, 2, 4):
            clip = torch.randn(batch, 8, 3, 5, dtype=torch.double)
            dino = torch.randn_like(clip)
            descriptors = F.adaptive_avg_pool2d(clip, 1).view(batch, 8)
            normalized = F.normalize(descriptors, dim=1)
            weights = (normalized @ normalized.t() / 0.05).softmax(-1)
            prototype = F.normalize(weights @ descriptors, dim=1).view(batch, 8, 1, 1)
            semantic = ((F.normalize(clip, dim=1) * prototype).sum(1, keepdim=True) / 0.05).sigmoid()
            normalized_dino = F.normalize(dino, dim=1).view(batch, 8, 15)
            affinity = (normalized_dino.transpose(1, 2).bmm(normalized_dino) / 0.05).softmax(-1)
            refined = affinity.bmm(semantic.view(batch, 15, 1)).view(batch, 1, 3, 5)
            module = CMCA().double()
            module.capture_vis = True
            actual_clip, actual_dino = module(clip, dino)
            torch.testing.assert_close(module.vis_maps['bscm'], semantic)
            torch.testing.assert_close(module.vis_maps['sapm'], refined)
            torch.testing.assert_close(actual_clip, clip * (1 + refined))
            torch.testing.assert_close(actual_dino, dino * (1 + refined))
            self.assertEqual(sum(p.numel() for p in module.parameters()), 0)

    def test_gradients_and_spatial_propagation(self):
        module = CMCA(semantic_temperature=0.5, structural_temperature=0.5).double()
        inputs = tuple(torch.randn(2, 3, 2, 2, dtype=torch.double, requires_grad=True) for _ in range(2))
        self.assertTrue(torch.autograd.gradcheck(module, inputs, atol=1e-4))
        uniform_dino = torch.ones(1, 3, 2, 2)
        response = torch.tensor([[[[0., 1.], [0., 1.]]]])
        torch.testing.assert_close(SAPM()(response, uniform_dino), torch.full_like(response, 0.5))

    def test_zero_and_invalid_inputs(self):
        zeros = torch.zeros(1, 2, 2, 3)
        result = CMCA()(zeros, zeros)
        self.assertTrue(all(torch.isfinite(value).all() for value in result))
        with self.assertRaises(ValueError):
            CMCA()(zeros, zeros[:, :, :, :1])
        with self.assertRaises(ValueError):
            BSCM(0)

    def test_losses_ignore_without_target_mutation(self):
        output = torch.randn(1, 3, 2, 2, requires_grad=True)
        target = torch.tensor([[[0, 1], [255, 2]]])
        original = target.clone()
        loss = dice_loss(output, target) + get_seg_loss(output, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        torch.testing.assert_close(target, original)
        changed = output.detach().clone()
        changed[:, :, 1, 0] = 100
        torch.testing.assert_close(dice_loss(output.detach(), target), dice_loss(changed, target))
        ignored = torch.full_like(target, 255)
        self.assertEqual(float(dice_loss(output, ignored) + get_seg_loss(output, ignored)), 0)
        for label in (0, 1):
            self.assertTrue(torch.isfinite(get_seg_loss(output, torch.full_like(target, label))))

    def test_configuration_paths_are_project_relative(self):
        cfg = load_config('configs/coco_attn_reg.yaml')
        self.assertTrue(Path(cfg.dataset.name_list_dir).is_absolute())
        self.assertEqual(cfg.dataset.num_classes, 81)
        self.assertEqual(cfg.cmca.start_iter, 10000)


class SyntheticDINO(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(3, 32, kernel_size=14, stride=14)

    def forward_features(self, image):
        return {'x_norm_patchtokens': F.normalize(self.projection(image).flatten(2).transpose(1, 2), dim=-1)}


class NetworkTests(unittest.TestCase):
    def make_model(self, dataset, device='cpu'):
        from clip.model import CLIP
        from cmca.model_attn_aff_voc import CMCANet as VOCNet
        from cmca.model_attn_aff_coco import CMCANet as COCONet
        clip_model = CLIP(embed_dim=32, image_resolution=32, vision_layers=3,
                          vision_width=64, vision_patch_size=16, context_length=77,
                          vocab_size=49408, transformer_width=32, transformer_heads=4,
                          transformer_layers=2).to(device)
        dino = SyntheticDINO()
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        repo = Path(temp.name)
        (repo / 'hubconf.py').write_text('# synthetic test repository')
        checkpoint = repo / 'dino.pth'
        torch.save(dino.state_dict(), checkpoint)
        model_class = VOCNet if dataset == 'voc' else COCONet
        with patch('cmca.model.clip.load', return_value=(clip_model, None)), patch('torch.hub.load', return_value=dino):
            model = model_class(embedding_dim=32, in_channels=[64] * 4, dino_fts_dim=32,
                                decoder_layers=1, device=device, dino_repo_path=str(repo),
                                dino_checkpoint=str(checkpoint), cmca_start_iter=2)
        return model

    def test_voc_coco_inference_shapes_and_freezing(self):
        for dataset, classes in [('voc', 21), ('coco', 81)]:
            model = self.make_model(dataset).eval()
            model.cmca.capture_vis = True
            with torch.no_grad():
                result = model(torch.randn(2, 3, 35, 50), mode='val')
            self.assertEqual(tuple(result[0].shape), (2, classes, 2, 3))
            self.assertEqual(result[0].shape, result[1].shape)
            self.assertIsNone(result[2])
            self.assertEqual(set(model.cmca.vis_maps), {'bscm', 'sapm'})
            self.assertEqual(model.iter_num, 0)
            self.assertIsNone(model.grad_cam)
            model.train()
            self.assertFalse(model.encoder.training)
            self.assertFalse(model.dino_encoder.training)
            self.assertTrue(all(not p.requires_grad for p in model.encoder.parameters()))

    def test_checkpoint_strict_loading_and_ddp_prefix(self):
        model = self.make_model('voc')
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'model.pth'
            state = {'module.' + key: value for key, value in model.state_dict().items()}
            state['module.encoder.visual.positional_embedding_new'] = torch.zeros(7, 64)
            torch.save({'state_dict': state}, checkpoint)
            result = load_checkpoint(model, checkpoint)
            self.assertFalse(result.missing_keys or result.unexpected_keys)
            state.pop(next(iter(state)))
            torch.save(state, checkpoint)
            with self.assertRaises(RuntimeError):
                load_checkpoint(model, checkpoint)

    @unittest.skipUnless(torch.cuda.is_available(), 'Training requires CUDA')
    def test_training_validation_save_and_inference_entry_points(self):
        from PIL import Image
        from omegaconf import OmegaConf
        from cmca import training, inference
        for dataset, foreground in [('voc', 20), ('coco', 80)]:
            model = self.make_model(dataset, device='cuda')
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                lists = root / 'lists'
                lists.mkdir()
                cfg = load_config('configs/{}_attn_reg.yaml'.format(dataset))
                cfg.dataset.root_dir = str(root)
                cfg.dataset.name_list_dir = str(lists)
                cfg.dataset.crop_size = 32
                cfg.dataset.rescale_range = [1., 1.]
                cfg.train.split = 'train'
                cfg.train.samples_per_gpu = 2
                cfg.train.max_iters = 1
                cfg.train.eval_iters = 1
                cfg.train.log_iters = 1
                cfg.runtime.num_workers = 0
                cfg.clip_init.resize_long = 48
                cfg.work_dir.ckpt_dir = str(root / 'checkpoints')
                cfg.work_dir.tb_logger_dir = str(root / 'logs')
                Path(cfg.work_dir.ckpt_dir).mkdir()
                class_labels = {}
                for split, count in [('train', 2), ('val', 1)]:
                    keys = []
                    for number in range(1, count + 1):
                        key = '{}_{:04d}'.format(split, number) if dataset == 'voc' else 'COCO_{}2014_{:012d}'.format(split, number)
                        keys.append(key)
                        class_labels[key] = np.eye(foreground, dtype=np.float32)[0]
                        folder = root / 'JPEGImages' if dataset == 'voc' else root / 'JPEGImages' / split / (split + '2014')
                        folder.mkdir(parents=True, exist_ok=True)
                        image_path = folder / (key + '.jpg')
                        Image.fromarray(np.random.default_rng(number).integers(0, 255, (32, 48, 3), dtype=np.uint8)).save(image_path)
                        if split == 'val':
                            labels_folder = root / 'SegmentationClass' if dataset == 'voc' else root / 'SegmentationClass' / split
                            labels_folder.mkdir(parents=True, exist_ok=True)
                            name = key if dataset == 'voc' else key.split('_')[-1]
                            label = np.zeros((32, 48), dtype=np.uint8)
                            label[:, 24:] = 1
                            Image.fromarray(label).save(labels_folder / (name + '.png'))
                    (lists / (split + '.txt')).write_text('\n'.join(keys) + '\n')
                np.save(lists / 'cls_labels_onehot.npy', class_labels)
                # No training segmentation masks or VOC XML annotations exist.
                self.assertFalse((root / 'SegmentationClassAug').exists())
                with patch.object(training, 'VOCNet' if dataset == 'voc' else 'COCONet', return_value=model):
                    self.assertTrue(training.train(cfg, dataset=dataset))
                checkpoint = Path(cfg.work_dir.ckpt_dir) / 'cmca_iter_1.pth'
                self.assertTrue(checkpoint.is_file())
                config_file = root / 'config.yaml'
                OmegaConf.save(cfg, str(config_file))
                arguments = ['inference', '--config', str(config_file), '--model_path', str(checkpoint),
                             '--input', str(image_path), '--work_dir', str(root / 'output'), '--scales', '1.0']
                with patch.object(sys, 'argv', arguments), patch('cmca.model_attn_aff_{}.CMCANet'.format(dataset), return_value=model):
                    inference.main(default_dataset=dataset)
                prediction_file = root / 'output' / 'val' / 'prediction' / (image_path.stem + '.png')
                with Image.open(prediction_file) as prediction:
                    self.assertEqual(prediction.size, (48, 32))
                self.assertTrue((root / 'output' / 'val' / 'logit' / (image_path.stem + '.npy')).is_file())
            model.grad_cam.activations_and_grads.release()
            del model
            torch.cuda.empty_cache()

    @unittest.skipUnless(torch.cuda.is_available(), 'Training CAM generation requires CUDA')
    def test_real_cam_par_losses_backward_and_warmup(self):
        for dataset, foreground in [('voc', 20), ('coco', 80)]:
            model = self.make_model(dataset, device='cuda').train()
            model.cmca.capture_vis = True
            # Use a nonsquare grid to catch token reshaping and affinity errors.
            image = torch.randn(2, 3, 32, 48, device='cuda')
            labels = torch.zeros(2, foreground)
            labels[:, 0] = 1
            optimizer = torch.optim.AdamW(model.get_param_groups()[3], lr=1e-4)
            weight_before = model.decoder.linear_pred.weight.detach().clone()
            for iteration in (1, 3):
                optimizer.zero_grad()
                clip_logits, dino_logits, pseudo, affinity = model(image, cls_labels=labels, iteration=iteration)
                self.assertEqual(tuple(pseudo.shape), (2, 32, 48))
                self.assertTrue(((pseudo == 0) | (pseudo == 1)).all())
                self.assertEqual(bool(model.cmca.vis_maps), iteration > 2)
                clip_full = F.interpolate(clip_logits, size=pseudo.shape[-2:], mode='bilinear', align_corners=False)
                dino_full = F.interpolate(dino_logits, size=pseudo.shape[-2:], mode='bilinear', align_corners=False)
                fused = 0.5 * (clip_full + dino_full)
                target_affinity = torch.eye(affinity.shape[-1], device='cuda')[None].expand_as(affinity)
                aff_loss, _, _ = get_aff_loss(affinity, target_affinity)
                loss = (get_seg_loss(fused, pseudo) + dice_loss(fused, pseudo) + 0.1 * aff_loss
                        + 0.1 * get_seg_loss(clip_full, dino_full.detach().argmax(1))
                        + 0.1 * get_seg_loss(dino_full, clip_full.detach().argmax(1)))
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for module in (model.decoder, model.decoder_fts_fuse, model.dino_decoder_fts_fuse):
                    self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in module.parameters()))
                self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
                optimizer.step()
            self.assertFalse(torch.equal(weight_before, model.decoder.linear_pred.weight))
            model.eval()
            with torch.no_grad():
                model(image, mode='val')
            self.assertEqual(model.iter_num, 3)
            model.grad_cam.activations_and_grads.release()
            del model
            torch.cuda.empty_cache()


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
