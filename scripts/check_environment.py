"""Check dependencies and configured assets without loading large models."""
import argparse
import importlib
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/voc_attn_reg.yaml')
    parser.add_argument('--skip-data', action='store_true')
    parser.add_argument('--crf', action='store_true')
    args = parser.parse_args()
    failures = []
    for name in ['torch', 'torchvision', 'numpy', 'PIL', 'cv2', 'omegaconf',
                 'imageio.v2', 'matplotlib', 'sklearn', 'scipy', 'tqdm',
                 'ftfy', 'regex', 'lxml', 'ttach', 'tensorboard', 'timm'] + (['pydensecrf'] if args.crf else []):
        try:
            module = importlib.import_module(name)
            print('OK dependency', name, getattr(module, '__version__', ''))
        except Exception as exc:
            failures.append('{}: {}'.format(name, exc))
    try:
        import torch
        print('CUDA available:', torch.cuda.is_available())
        if not torch.cuda.is_available():
            print('Training requires CUDA; CPU segmentation inference is available.')
        from cmca.runtime import load_config
        cfg = load_config(args.config)
        paths = [Path(cfg.clip_init.clip_pretrain_path), Path(cfg.dino_init.repo_path) / 'hubconf.py', Path(cfg.dino_init.checkpoint)]
        if not args.skip_data:
            paths += [Path(cfg.dataset.root_dir) / 'JPEGImages',
                      Path(cfg.dataset.name_list_dir) / (cfg.train.split + '.txt'),
                      Path(cfg.dataset.name_list_dir) / 'cls_labels_onehot.npy']
        for path in paths:
            if path.exists():
                print('OK asset', path)
            else:
                failures.append('Missing asset: {}'.format(path))
    except Exception as exc:
        failures.append('Configuration: {}'.format(exc))
    for failure in failures:
        print('FAIL', failure)
    return int(bool(failures))


if __name__ == '__main__':
    sys.exit(main())
