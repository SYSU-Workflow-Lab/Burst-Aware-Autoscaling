import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class LightningPredictor(pl.LightningModule):
    def __init__(self, enc_number = 1, hidden_num = 30, n_output = 1):
        super(LightningPredictor, self).__init__()
        self.forward1 = nn.Linear(enc_number, hidden_num)
        self.forward2 = nn.Linear(hidden_num, hidden_num)
        self.forward3 = nn.Linear(hidden_num, n_output)
        self.double()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(dim=-1)
        trainX = torch.transpose(torch.stack(x), 0, 1)
        y_hat = self(trainX)
        loss = F.mse_loss(y_hat, y)
        return loss

    def forward(self, x):
        # x = x.float()
        xx1 = self.forward1(x)
        x1 = nn.Sigmoid()(xx1)
        xx2 = self.forward2(x1)
        x2 = nn.Sigmoid()(xx2)
        xx3 = self.forward3(x2)
        x3 = nn.Sigmoid()(xx3)
        return x3