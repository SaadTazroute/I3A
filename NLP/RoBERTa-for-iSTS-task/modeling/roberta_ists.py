# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from fairseq.models.roberta import RobertaModel

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

class RobertaISTS(torch.nn.Module):
    def __init__(self, num_classes, dropout_rate, hidden_neurons=2048):
        super(RobertaISTS, self).__init__()

        #self.roberta = torch.hub.load(model='roberta.large')
        self.roberta = RobertaModel.from_pretrained('/home/phillyflingo/PycharmProjects/PSTALN/RoBERTa-for-iSTS-task/roberta.large',checkpoint_file='model.pt')
        #self.roberta = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')    # Download model and configuration from S3 and cache.
        #self.roberta = BertModel.from_pretrained('bert-base-uncased')

        #self.roberta = .from_pretrained('/path/to/roberta.large', checkpoint_file='model.pt')

        for param in self.roberta.parameters():
            param.requires_grad = False

        self.linear1 = nn.Linear(in_features=1024, out_features=hidden_neurons, bias=True)
        self.linear2 = nn.Linear(in_features=hidden_neurons, out_features=1, bias=True)
        self.linear3 = nn.Linear(in_features=1024, out_features=hidden_neurons, bias=True)
        self.linear4 = nn.Linear(in_features=hidden_neurons, out_features=num_classes, bias=True)
        print('XXXXXXXXXXXXXXXXXXXXXxx')
        print(num_classes)
        print('XXXXXXXXXXXXXXXXXXXXXxx')
        self.dropout1 = nn.Dropout(p=dropout_rate, inplace=False)
        self.dropout2 = nn.Dropout(p=dropout_rate, inplace=False)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):

        features = self.roberta.extract_features(x)
        x = torch.mean(features, 1, keepdim=False)
        x = self.dropout1(x)

        #STS value regression
        x1 = self.relu1(self.linear1(x))
        x1 = self.dropout2(x1)
        out1 = self.linear2(x1)[0]

        #Classification  layer
        x2 = self.relu2(self.linear3(x))
        out2 = F.log_softmax(self.linear4(x2), dim=1)

        return out1, out2
