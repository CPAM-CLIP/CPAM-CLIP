import os
import argparse
import clip_segmentor
import custom_datasets

from mmengine.config import Config
from mmengine.runner import Runner

import sys
import random

hyper_params = [[{"P":0.14,
    "T":0.13,
    "kl_sizes":[
        0.7950609569058564,
        0.5763927164676379,
        0.47685060820688063,
        1.2709011643727646,
    ]}],
[{"P":0.14,
    "T":0.12,
    "kl_sizes":[
        1.1432482214250812,
        0.6352359682851626,
        1.2880490568637535,
        0.25805479974968226,
    ]}],
[{"P":0.12,
    "T":0.1,
    "kl_sizes":[
        0.4552496800380126,
        0.5271273350336941,
        1.4777846902346798,
        0.8473469862457899,
        0.3564148352929275,
    ]}],
[{"P":0.14,
    "T":0.11,
    "kl_sizes":[
        0.47047833997318167,
        1.2939338617083993,
        0.9310561998912192,
        0.3972909692606767,
    ]}],
[{"P":0.13,
    "T":0.11,
    "kl_sizes":[
        0.48384100443534883,
        0.6178362776524873,
        1.0584789241987287,
    ]}],
[{"P":0.13,
    "T":0.11,
    "kl_sizes":[
        0.4885085983402661,
        0.3538529309407541,
        0.4945043779593715,
        0.2775056074287384,
        1.03810162499402,
        1.4592184312166192,
        1.2693789940940736,
    ]}],
[{"P":0.14,
    "T":0.07,
    "kl_sizes":[
        1.3968447313341605,
        0.3031476649645178,
        0.3936707295057206,
        1.419740968290217,
        1.4293314667440251,
    ]}],
[{"P":0.11,
    "T":0.1,
    "kl_sizes":[
        0.25752965288615215,
        1.3761542948879728,
        0.8213494548471293,
        1.3863503553969223,
        0.7483765979227761,
        1.117562709254153,
    ]}],
    [{"P":0.13,
    "T":0.07,
    "kl_sizes":[
        0.42853711967396085,
        0.5045740282605191,
        0.9806360192372824,
        1.0528539255764533,
        0.4834588119300615,
    ]}],
[{"P":0.14,
    "T":0.07,
    "kl_sizes":[
        1.3188280517406283,
        0.28370876235154274,
        0.47531373377481084,
        1.0052198329963882,
    ]}],
[{"P":0.13,
    "T":0.08,
    "kl_sizes":[
        0.4357645077379804,
        0.8724734860484172,
        0.8154229952868777,
    ]}],
[{"P":0.13,
    "T":0.1,
    "kl_sizes":[
        0.46555903253573455,
        1.415936305310633,
        0.8265335699031542,
        0.5174682972295298,
    ]}],
    [{"P":0.12,
    "T":0.1,
    "kl_sizes":[
        0.683272346511407,
        0.40889253539631826,
        1.0144424595180013,
        1.3851729310702938,
        0.5147535372399998,
        0.9080954116046096,
        0.472658350610092,
    ]}],
[{"P":0.14,
    "T":0.11,
    "kl_sizes":[
        0.49406271085208825,
        1.0859272394061765,
        0.5669889242423238,
        0.8287460220330667,
    ]}],
    [{"P":0.14,
    "T":0.07,
    "kl_sizes":[
        1.3597391446078138,
        0.42804966818896883,
        0.4604090774790615,
        0.4860316303749122,
        1.0363783782174578,
        0.6062500348031509,
        1.2566648487779861,
    ]}],
[{"P":0.14,
    "T":0.12,
    "kl_sizes":[
        0.7358189797507579,
        1.2426568078117906,
        0.6386384974619788,
        0.9346508913029936,
        0.7077581840073998,
    ]}],
[{"P":0.13,
    "T":0.12,
    "kl_sizes":[
        1.1198909147171134,
        0.7402831648471757,
        0.7157766712728635,
        0.6218544204630956,
        0.6515715283890697,
        1.4606206409837,
        0.997731633964196,
    ]}]]


# hyper_params = [
# [{"P":0.14,
#     "T":0.17,
#     "kl_sizes":[
#         0.9749851923788148,
#         0.804973056633385,
#         0.6138345976796581
#     ]}]]



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
    
    
    parser.add_argument('--index', default=0, type=int, required=True, help='Temperature 2 value')

    
    
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