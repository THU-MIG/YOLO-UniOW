# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp

import torch
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_state_dict

from mmyolo.registry import RUNNERS
from mmyolo.utils import is_metainfo_lower


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    # replace the ${key} with the value of cfg.key
    # cfg = replace_cfg_vals(cfg)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        if args.config.startswith('projects/'):
            config = args.config[len('projects/'):]
            config = config.replace('/configs/', '/')
            cfg.work_dir = osp.join('./work_dirs', osp.splitext(config)[0])
        else:
            cfg.work_dir = osp.join('./work_dirs',
                                    osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # Determine whether the custom metainfo fields are all lowercase
    is_metainfo_lower(cfg)

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner._train_loop = runner.build_train_loop(
            runner._train_loop)  # type: ignore

    # `build_optimizer` should be called before `build_param_scheduler`
    #  because the latter depends on the former
    runner.optim_wrapper = runner.build_optim_wrapper(runner.optim_wrapper)
    # Automatically scaling lr by linear scaling rule
    runner.scale_lr(runner.optim_wrapper, runner.auto_scale_lr)

    if runner.param_schedulers is not None:
        runner.param_schedulers = runner.build_param_scheduler(  # type: ignore
            runner.param_schedulers)  # type: ignore

    if runner._val_loop is not None:
        runner._val_loop = runner.build_val_loop(
            runner._val_loop)  # type: ignore
    # TODO: add a contextmanager to avoid calling `before_run` many times
    runner.call_hook('before_run')

    # initialize the model weights
    runner._init_model_weights()

    # make sure checkpoint-related hooks are triggered after `before_run`
    if not runner._has_loaded:
        checkpoint = torch.load(runner._load_from, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        if is_model_wrapper(runner.model):
            model = runner.model.module
        else:
            model = runner.model

        state_dict = {k: v for k, v in state_dict.items() if 'text_model' not in k}
        if "embeddings" in state_dict:
            # update embeddings in checkpoint before loading
            state_dict['embeddings'] = model.update_embeddings(state_dict['embeddings'])
            print_log(f"Updated Embeddings to {state_dict['embeddings'].shape}" , logger='current', level=logging.INFO)
        checkpoint['state_dict'] = state_dict

        runner.call_hook('after_load_checkpoint', checkpoint=checkpoint)
        load_state_dict(model, state_dict)

        print_log(
                f'Load checkpoint from {runner._load_from}',
                logger='current',
                level=logging.INFO)

        runner._has_loaded = True
    

    # Initiate inner count of `optim_wrapper`.
    runner.optim_wrapper.initialize_count_status(
        runner.model,
        runner._train_loop.iter,  # type: ignore
        runner._train_loop.max_iters)  # type: ignore

    # Maybe compile the model according to options in runner.cfg.compile
    # This must be called **AFTER** model has been wrapped.
    runner._maybe_compile('train_step')

    model = runner.train_loop.run()  # type: ignore
    runner.call_hook('after_run')


if __name__ == '__main__':
    main()
