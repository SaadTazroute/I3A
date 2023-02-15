# encoding: utf-8

import logging
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
import torch

def inference(
        cfg,
        model,
        val_loader
):
    device_type = cfg.MODEL.DEVICE
    device = torch.device(device_type)
    log_period = cfg.SOLVER.LOG_PERIOD
    logger = logging.getLogger("model.inference")
    logger.info("Start inferencing")

    gold_val = []
    gold_exp = []
    pred_val = []
    pred_exp = []
    model.eval()
    model.to(device)

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, value, explanation = data[0].to(device), data[1].to(device), data[2].to(device)
            out1, out2 = model(inputs)
            _, predicted = torch.max(out2, 1)
            gold_val.append(value.item())
            gold_exp.append(explanation.item())
            pred_val.append(out1.item())
            print(out1.item())
            pred_exp.append(predicted.item())
            print(predicted.item())
            if i % log_period == log_period -1:
                logger.info('Progress [%d/%d]' %
                        (i + 1, len(val_loader)))

    logger.info('| F1 score for explanations: %.3f' % f1_score(gold_exp, pred_exp, average='micro'))
    round_to_whole = [round(num) for num in pred_val]
    logger.info('| F1 score for values: %.3f' % f1_score(gold_val, round_to_whole, average='micro'))
