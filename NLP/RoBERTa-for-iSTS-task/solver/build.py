# encoding: utf-8

import torch


def make_optimizer(cfg, model):
    optimizer = torch.optim.Adam(model.parameters(), lr= cfg.SOLVER.BASE_LR)
    return optimizer
