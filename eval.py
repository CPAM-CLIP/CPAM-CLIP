import os
import argparse
import clip_segmentor
import custom_datasets

from mmengine.config import Config
from mmengine.runner import Runner

import sys
import random

# hyper_params = [[{"P":0.2,
#     "T":0.09,
#     "kl_sizes":[
#         0.8740030888922894,
#         0.7883244540065957,
#         0.39557169903567346,
#         0.5177720690896945,
#         0.30758696333316216,
#         0.8996076111887488,
#         0.6497661579551135
#     ]}],
# [{"P":0.15,
#     "T":0.07,
#     "kl_sizes":[
#         0.5783392177151874,
#         0.5991643473219477,
#         0.7368725429045726
#     ]}],
# [{"P":0.17,
#     "T":0.13,
#     "kl_sizes":[
#         0.6937026133853298,
#         0.8082457515071951,
#         0.7833581972956898
#     ]}],
# [{"P":0.1,
#     "T":0.09,
#     "kl_sizes":[
#         0.9619752097641435,
#         0.22657571914780916,
#         0.7554307834821781,
#         0.5685297506730932,
#         0.5499091291286305,
#         0.2879023681203726,
#         0.5887509434697407
#     ]}],
# [{"P":0.1,
#     "T":0.08,
#     "kl_sizes":[
#         0.77101550763397,
#         0.4113607995310389,
#         0.31481733078379515,
#         0.7858322766373605,
#         0.9035387418347414,
#         0.32070520766767263
#     ]}],
# [{"P":0.14,
#     "T":0.12,
#     "kl_sizes":[
#         0.7752192035991311,
#         0.925508381308105,
#         0.47315830679538073,
#         0.6482341965121091,
#         0.6866127396912376
#     ]}],
# [{"P":0.14,
#     "T":0.07,
#     "kl_sizes":[
#         0.8145020068876604,
#         0.25629445427335873,
#         0.8860959710168734,
#         0.6133828571050156
#     ]}],
# [{"P":0.13,
#     "T":0.1,
#     "kl_sizes":[
#         0.8751835918603615,
#         0.26889281520838515,
#         0.9456278763885965
#     ]}],
# [{"P":0.14,
#     "T":0.18,
#     "kl_sizes":[
#         0.9414559755432428,
#         0.8704740441581599,
#         0.9504402830371943
#     ]}],
# [{"P":0.16,
#     "T":0.15,
#     "kl_sizes":[
#         0.7912663209195689,
#         0.4931457467811954,
#         0.8553594358423642,
#         0.7155944058359154,
#         0.9638857251271546
#     ]}],
# [{"P":0.13,
#     "T":0.07,
#     "kl_sizes":[
#         0.9523356040383827,
#         0.9034917759081119,
#         0.7401601797971487,
#         0.7735051599217577
#     ]}],
# [{"P":0.14,
#     "T":0.17,
#     "kl_sizes":[
#         0.9749851923788148,
#         0.804973056633385,
#         0.6138345976796581
#     ]}],
# [{"P":0.13,
#     "T":0.1,
#     "kl_sizes":[
#         0.688305445565408,
#         0.6662089668832931,
#         0.9707325102396572
#     ]}],
# [{"P":0.12,
#     "T":0.1,
#     "kl_sizes":[
#         0.8454323043900673,
#         0.40993241171902584,
#         0.4122865071367582,
#         0.5569728318430855,
#         0.7573063359880214
#     ]}],
# [{"P":0.07,
#     "T":0.19,
#     "kl_sizes":[
#         0.7118267361713925,
#         0.9567144144448341,
#         0.5014505630922095,
#         0.7308531472517156
#     ]}],
# [{"P":0.13,
#     "T":0.13,
#     "kl_sizes":[
#         0.8818000732589452,
#         0.6986692183215663,
#         0.921282873960587,
#         0.31929478988312043,
#         0.5740025086474836,
#         0.9426262948605687
#     ]}]]


hyper_params = [
[{"P":0.14,
    "T":0.17,
    "kl_sizes":[
        0.9749851923788148,
        0.804973056633385,
        0.6138345976796581
    ]}]]



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
    
    
    parser.add_argument('--index', type=int, required=True, help='Temperature 2 value',default=0)

    
    
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
    # cfg.model['T'] = random.randrange(5, 25) * 0.01 #[0.05 - 0.3]
    # cfg.model['P'] = random.randrange(5, 25) * 0.01 #[0.05 - 0.3]
    # size_len = random.randrange(3, 8)
    index = args.index
    cfg.model['T'] = hyper_params[index][0]["T"]
    cfg.model['P'] = hyper_params[index][0]["P"]
    cfg.model['kl_sizes'] = hyper_params[index][0]["kl_sizes"]
    
    trigger_visualization_hook(cfg, args)

    runner = Runner.from_cfg(cfg)
    runner.test()

if __name__ == '__main__':
    main()