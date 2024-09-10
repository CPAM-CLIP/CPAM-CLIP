import os
import random
import argparse


if __name__=="__main__":
    for _ in range(300):
        os.system(f"python eval.py --config ./configs/cfg_voc21.py")
            