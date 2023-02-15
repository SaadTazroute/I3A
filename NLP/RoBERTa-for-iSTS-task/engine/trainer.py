# encoding: utf-8

import logging
import numpy as np
import torch
import datetime


def do_train(
        cfg,
        model,
        train_loader,
        optimizer,
        losses,
):

    output_dir = cfg.OUTPUT_DIR
    output_dir = '/home/phillyflingo/PycharmProjects/PSTALN/RoBERTa-for-iSTS-task/output-bert'

    device = torch.device('cuda:0')

    epochs = cfg.SOLVER.MAX_EPOCHS
    log_period = cfg.SOLVER.LOG_PERIOD
    
    logger = logging.getLogger("model.train")
    logger.info("Start training")


    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_loss1 = 0.0
        running_loss2= 0.0

        partial_loss1 = 0.0
        partial_loss2 = 0.0
        print("epoch actuelle:",epoch)
        #print("loss actuelle :", loss)
        for i, data in enumerate(train_loader, 0):
            inputs, value, explanation = data[0].to(device), data[1].to(device), data[2].to(device)

            # forward + backward + optimize
            out1 = model(inputs[0])[0].to(device)
            out2 = model(inputs[0])[1].to(device)

            explanation = explanation.type(torch.LongTensor).to(device)
            loss1 = losses[0](out1, value)
            loss2 = losses[1](out2, explanation)
            loss = loss1 + loss2


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_loss1 += loss1.item()
            running_loss2 += loss2.item()
            partial_loss1 += loss1.item()
            partial_loss2 += loss2.item()
            if i % log_period == log_period -1:
                logger.info('EPOCH: [%d/%d] BATCHES [%d/%d] loss summed: %.3f loss MSE: %.3f loss NLLL: %.3f' %
                        (epoch + 1, epochs, i + 1, len(train_loader) ,(partial_loss1 + partial_loss2) / log_period, partial_loss1 / log_period, partial_loss2 / log_period))
                partial_loss1 = 0.0
                partial_loss2 = 0.0


        logger.info('EPOCH: [%d] FINISHED loss summed: %.3f loss MSE: %.3f loss NLLL: %.3f' %
                    (epoch + 1, running_loss / len(train_loader), running_loss1 / len(train_loader), running_loss2 / len(train_loader)))
        running_loss = 0.0
        running_loss1 = 0.0
        running_loss2 = 0.0


        logger.info('Finished Training')
        logger.info('Saving model ...')
        output_filename = output_dir + '/' + datetime.datetime.now().strftime("%d%m%Y%H%M%S") + '_model.pt'
        torch.save(model.state_dict(), output_filename) 
        logger.info('Model saved as :' + output_filename)
