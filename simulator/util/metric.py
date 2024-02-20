import numpy as np

def SMAPE(pred, true):
    if isinstance(pred,np.ndarray):
        if len(pred.shape)>=2:
            return 100 / pred.shape[1] / pred.shape[0] * np.sum(2 * np.abs(true - pred) / (np.abs(pred) + np.abs(true)))
        else:
            return 100 / pred.shape[0] * np.sum(2 * np.abs(true - pred) / (np.abs(pred) + np.abs(true)))
    else:
        return 100 * np.sum(2 * np.abs(true - pred) / (np.abs(pred) + np.abs(true)))

def MSE(pred, true):
    return np.mean((pred - true) ** 2)

import torch
import torch.nn as nn
import numpy as np
# 核心组件
class QuantileLoss(nn.Module):
    ## From: https://medium.com/the-artificial-impostor/quantile-regression-part-2-6fdbc26b2629

    def __init__(self, quantiles):
        ##takes a list of quantiles
        super().__init__()
        self.quantiles = quantiles

    def numpy_normalised_quantile_loss(self, y_pred, y):
        acc = 0
        acc += self.normalized_quantile_loss(y_pred,y,quantile=self.quantiles[0])
        acc += self.normalized_quantile_loss(y_pred,y,quantile=self.quantiles[-1])
        return acc / 2

    def normalized_quantile_loss(self, y_pred, y, quantile=0.5):
        """Computes normalised quantile loss for numpy arrays.
        Uses the q-Risk metric as defined in the "Training Procedure" section of the
        main TFT paper.
        ref:https://github.com/stevinc/Transformer_Timeseries/blob/master/utils.py
        Args:
          y: Targets
          y_pred: Predictions
          quantile: Quantile to use for loss calculations (between 0 & 1)
        Returns:
          Float for normalised quantile loss.
        """
        assert quantile in self.quantiles
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()

        if len(y_pred.shape) == 3:
            if y_pred.shape[-1]>1: # quantile下提取数据
                ix = self.quantiles.index(quantile)
                y_pred = y_pred[..., ix]
            else: # 正常数据进行偏向估计
                y_pred = y_pred[...,0]
        elif len(y_pred.shape) == 2:
            if y_pred.shape[-1]>1: # quantile下提取数据
                ix = self.quantiles.index(quantile)
                y_pred = y_pred[..., ix]
            else: # 正常数据进行偏向估计
                y_pred = y_pred[...,0]
        elif len(y_pred.shape) == 1:
            ix = self.quantiles.index(quantile)
            y_pred = y_pred[ix]
        elif len(y_pred.shape) + len(y.shape) > 0:
            # 要么都等于0，不然就有问题
            raise Exception

        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        if len(y.shape)==3:
            y = y[...,0]

        prediction_underflow = y - y_pred
        weighted_errors = quantile * np.maximum(prediction_underflow, 0.) \
                          + (1. - quantile) * np.maximum(-prediction_underflow, 0.)

        quantile_loss = weighted_errors.mean()
        normaliser = np.abs(y).mean()

        return 2 * quantile_loss / normaliser

    def forward(self, preds, target):
        """
            preds为预测值,target为真实值
            其长度为(batch_size, time_length, num_features)
            其中preds的num_features=c_out，与quantile_list等长
            target在这里为一个值
        """
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[..., i].item()
            losses.append(max(q * errors, (q-1)*errors))
        loss = np.mean(losses)
        normaliser = np.abs(target).mean()
        return 2 * loss / normaliser
        