from pathlib import Path
import glob
import numpy as np
import torch

from mmengine.config import Config
from mmengine.runner import Runner


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Extract text features')
    parser.add_argument('--config', type=str)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--wildcard', type=str, default='object', help='Wildcard to extract features from')
    parser.add_argument('--save_path', type=str, default='embeddings', help='Save path for extracted features')
    parser.add_argument('--extract_tuned', action='store_true', help='Extract tuned wildcard embeddings')
    return parser.parse_args()


@torch.inference_mode()
def extract_feats(model, dataset=None, task=None, save_path='embeddings'):
    prompt = []
    prompt_path = f'data/OWOD/ImageSets/{dataset}/t{task}_known.txt'
    with open(prompt_path, 'r') as f:
        prompt = [line.strip() for line in f.readlines()]
    save_path = save_path / f'{dataset.lower()}_t{task}.npy'

    text_feats = model.backbone.forward_text([prompt]).squeeze(0).detach().cpu()
    print(f"Extracted text features from {dataset}/Task({task}):", text_feats.shape)

    np.save(save_path, text_feats.numpy())


@torch.inference_mode()
def extract_tuned_feats(config=None, ckpt=None, wildcard='object', save_path="embeddings"):
    # extract tuned wildcard embeddings from text encoder
    model_path = sorted(glob.glob(f'work_dirs/{Path(config).stem}/best*.pth'))[-1] if ckpt is None else ckpt
    tuned_feats = torch.load(model_path, map_location='cpu')['state_dict']['embeddings']
    print("Extracted tuned wildcard text features:", tuned_feats.shape)
    np.save(save_path / f'{wildcard.replace(" ", "_")}_tuned.npy', tuned_feats.numpy())


@torch.inference_mode()
def extract_wildcard_feats(model, wildcard='object', save_path='embeddings'):
    # extract wildcard embeddings from text encoder
    save_path = save_path / f'{wildcard.replace(" ", "_")}.npy'
    text_feats = model.backbone.forward_text([[wildcard]]).squeeze(0).detach().cpu()
    print("Extracted wildcard text features:", text_feats.shape)
    np.save(save_path, text_feats.numpy())


if __name__ == "__main__":
    args = parse_args()
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    if args.extract_tuned:
        extract_tuned_feats(args.config, args.ckpt, save_path=save_path)
    else:
        # init model
        cfg = Config.fromfile(args.config)
        cfg.work_dir = 'work_dirs/extract_feats'

        runner = Runner.from_cfg(cfg)
        runner.call_hook("before_run")
        runner.load_checkpoint(args.ckpt, map_location='cpu')
        model = runner.model.to('cuda')
        model.eval()

        # extract features
        extract_wildcard_feats(model, wildcard=args.wildcard, save_path=save_path)
        for i in range(1, 5):
            extract_feats(model, dataset='MOWODB', task=i, save_path=save_path)
            extract_feats(model, dataset='SOWODB', task=i, save_path=save_path)
            if i < 4:
                extract_feats(model, dataset='nuOWODB', task=i, save_path=save_path)