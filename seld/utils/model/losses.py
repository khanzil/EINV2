import torch
import torch.nn as nn

class Losses:
    def __init__(self, cfg):
        self.loss_type = cfg['train']['loss_type']
        self.loss_beta = cfg['train']['loss_beta']








