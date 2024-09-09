import os
import argparse
import clip_segmentor
import custom_datasets

from mmengine.config import Config
from mmengine.runner import Runner

import sys
import random

def parse_args():
    parser = argparse.ArgumentParser(
        description='SCLIP evaluation with MMSeg')
    parser.add_argument('--config', default='')
    # parser.add_argument('--work-dir', default='./work_logs/')
    parser.add_argument('--work-dir', default='./hyperparameter/')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show_dir',
        default='./outputs/',
        help='directory to save visualizaion images')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)

    
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            visualizer = cfg.visualizer
            visualizer['save_dir'] = args.show_dir
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    cfg.work_dir = args.work_dir
    # breakpoint()
    cfg.model['T'] = random.randrange(5, 25) * 0.01 #[0.05 - 0.3]
    cfg.model['P'] = random.randrange(5, 25) * 0.01 #[0.05 - 0.3]
    size_len = random.randrange(3, 8)
    cfg.model['kl_sizes'] = [ random.random() for _ in range(size_len)]
    
    trigger_visualization_hook(cfg, args)

    runner = Runner.from_cfg(cfg)
    runner.test()

if __name__ == '__main__':
    main()