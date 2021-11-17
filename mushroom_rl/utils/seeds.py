import random
import numpy as np
import torch


def fix_random_seed(seed, mdp=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if mdp is not None:
        try:
            mdp.seed(seed)
        except:
            pass