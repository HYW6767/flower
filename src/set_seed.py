#设置种子


import random
import numpy as np
import torch


def seedset(seeds):
    random.seed(seeds)
    np.random.seed(seeds)
    torch.manual_seed(seeds)