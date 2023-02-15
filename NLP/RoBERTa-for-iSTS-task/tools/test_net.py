# encoding: utf-8

import argparse
import os
import sys
from os import mkdir

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.tensorboard import SummaryWriter

sys.path.append('.')
from net_config import cfg
from data import make_data_loader
from engine.inference import inference
from modeling import build_model
from utils.logger import setup_logger
import torch
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
def main():
    parser = argparse.ArgumentParser(description="Roberta iSTS Inference")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("model", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)
    logger.propagate = False

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    model = build_model(cfg)
    writer = SummaryWriter('testtest')
    print(model)
    model.load_state_dict(torch.load(cfg.TEST.WEIGHT))
    val_loader = make_data_loader(cfg, cfg.DATASETS.TEST,is_train=False)
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print(cfg.DATASETS.TEST)
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    inference(cfg, model, val_loader)


if __name__ == '__main__':
    main()
