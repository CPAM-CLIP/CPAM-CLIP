import os
import random
import argparse


if __name__=="__main__":
    for i in range(16):
        os.system(f"python eval.py --config ./configs/cfg_context60.py --work-dir ./hyper-context --index {i}")
            