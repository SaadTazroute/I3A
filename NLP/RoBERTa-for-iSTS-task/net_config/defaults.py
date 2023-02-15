from yacs.config import CfgNode as CN
import torch
import numpy as np
# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

_C.MODEL.NUM_CLASSES = 6
_C.MODEL.DROPOUT = 0.25
_C.MODEL.HIDDEN_NEURONS = 1024

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training
_C.DATASETS.TRAIN = ("data/datasets/train.csv")
# List of the dataset names for testing
_C.DATASETS.TEST = ("data/datasets/test.csv")
_C.DATASETS.TEST_WA = ("data/datasets/STSint.testinput.answers-students.wa")

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "ADAM"
_C.SOLVER.MAX_EPOCHS = 200
_C.SOLVER.BASE_LR = 0.0001
_C.SOLVER.LOG_PERIOD = 100
_C.SOLVER.BATCH_SIZE = 1

_C.TEST = CN()
_C.TEST.BATCH_SIZE = 1
_C.TEST.WEIGHT = ""


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "output"
