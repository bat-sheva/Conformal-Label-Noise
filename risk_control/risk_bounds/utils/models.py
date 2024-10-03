"""
Code copied from https://github.com/YoungseogChung/calibrated-quantile-uq
"""
import abc
import os, sys
from copy import deepcopy

import matplotlib.pyplot
import tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import helper
from losses import batch_mse_loss, selective_net_loss, two_headed_qr_loss, batch_qr_loss, MetricLoss, \
    MetricCalibrationLoss

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from NNKit.models.model import vanilla_nn, selective_net_nn, binary_nn

"""
Define wrapper uq_model class
All uq models will import this class
"""


class Model:
    @property
    @abc.abstractmethod
    def name(self):
        pass


class PredictiveModel(Model, abc.ABC):
    def __init__(self):
        self._model = None
        self._optimizers = None
        self.loss_fn = None
        self.keep_training = True
        self.best_va_loss = np.inf
        self.best_va_model = None
        self.best_va_ep = 0
        self.done_training = False

    def predict(self, x, use_best_va_model=True):

        with torch.no_grad():
            if use_best_va_model and self.best_va_model is not None:
                self.best_va_model.eval()
                pred = self.best_va_model(x)
            else:
                self.model.eval()
                pred = self.model(x)

        return pred

    def loss(self, loss_fn, x, y, take_step, args, weights=None, index=None, is_val=False):
        self.optimizers.zero_grad()
        if self.keep_training:
            if take_step:
                self.model.train()
            else:
                self.model.eval()

            loss = loss_fn(self.model, y, x, self.device, args, weights=weights, index=index, is_val=is_val)

            if take_step:
                loss.backward()
                self.optimizers.step()
            loss = loss.item()
        else:
            loss = np.nan

        return loss

    def train(self, x_tr, y_tr, x_va, y_va, args, print_starting_to_train=True):
        if print_starting_to_train:
            print(f"starting to train: {self.name}")

        loader = DataLoader(helper.IndexedDataset(x_tr, y_tr),
                            shuffle=True,
                            batch_size=args.bs)
        va_losses = []
        # Loss function
        """ train loop """
        for ep in tqdm.tqdm(range(args.num_ep)):
            if self.done_training:
                break

            # Take train step
            for xi, yi, index in loader:
                self.calc_train_loss(self.loss_fn, xi, yi, take_step=True, args=args, index=index)
                # self.loss(self.loss_fn, xi, yi, take_step=True, args=args)

            va_losses += [self.update_va_loss(self.loss_fn, x_va, y_va, curr_ep=ep, num_wait=args.wait, args=args)]
        # matplotlib.pyplot.plot(va_losses)
        # matplotlib.pyplot.show()

    def calc_train_loss(self, loss_fn, x, y, take_step, args, index):
        self.loss(loss_fn, x, y, take_step=take_step, args=args, index=index)

    def calc_va_loss(self, loss_fn, x, y, args, weights=None):
        with torch.no_grad():
            return self.loss(loss_fn, x, y, take_step=False, args=args, weights=weights, is_val=True)

    def update_va_loss(self, loss_fn, x, y, curr_ep, num_wait, args, weights=None):
        with torch.no_grad():
            va_loss = self.calc_va_loss(loss_fn, x, y, args, weights=None)

        if self.keep_training:
            if va_loss < self.best_va_loss and curr_ep >= 30:
                self.best_va_loss = va_loss
                self.best_va_ep = curr_ep
                self.best_va_model = deepcopy(self.model)
            else:
                if curr_ep - self.best_va_ep > num_wait:
                    self.keep_training = False

        if not self.keep_training:
            self.done_training = True

        return va_loss

    def use_device(self, device):
        self.device = device
        self.best_va_model = self.best_va_model.to(device)

        if device.type == 'cuda':
            assert next(self.best_va_model.parameters()).is_cuda

    def print_device(self):
        device_list = []
        if next(self.best_va_model.parameters()).is_cuda:
            device_list.append('cuda')
        else:
            device_list.append('cpu')
        print(device_list)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def loss_fn(self):
        return self._loss_fn

    @loss_fn.setter
    def loss_fn(self, value):
        self._loss_fn = value

    @property
    def optimizers(self):
        return self._optimizers

    @optimizers.setter
    def optimizers(self, value):
        self._optimizers = value


class UncertaintyModel(Model, abc.ABC):

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def predict_uncertainty(self, x, use_best_va_model=True):
        pass

    @abc.abstractmethod
    def train(self, x_tr, y_tr, x_va, y_va, args):
        pass



class MetricSelectionModel(PredictiveModel):

    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout, lr, wd, device,
                 calc_metric, calc_calibrated_pred, desired_level, coeff, x_ca, y_ca, cal_pred,
                 miscoverage_level, train_preds, val_preds):
        super().__init__()
        self.device = device
        self.dropout = dropout
        self.input_size = input_size
        self.model = binary_nn(input_size=input_size, output_size=output_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                dropout=dropout).to(device)
        self.optimizers = torch.optim.Adam(self.model.parameters(),
                                           lr=lr, weight_decay=wd)
        self.loss_fn = MetricLoss(calc_metric, calc_calibrated_pred, desired_level, coeff, x_ca, y_ca, cal_pred, miscoverage_level, train_preds, val_preds).loss

    @property
    def name(self):
        return "Metric Selection Model"


class MetricCalibrationModel(PredictiveModel):

    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout, lr, wd, device,
                 calc_metric, calc_calibrated_pred_for_metric, desired_level, coeff, x_ca, y_ca, cal_pred,
                 miscoverage_level, train_preds, val_preds):
        super().__init__()
        self.device = device
        self.dropout = dropout
        self.input_size = input_size
        self.model = vanilla_nn(input_size=input_size, output_size=output_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                dropout=dropout).to(device)
        self.optimizers = torch.optim.Adam(self.model.parameters(),
                                           lr=lr, weight_decay=wd)
        self.loss_fn = MetricCalibrationLoss(calc_metric, calc_calibrated_pred_for_metric, desired_level, coeff, x_ca, y_ca, cal_pred, miscoverage_level, train_preds, val_preds).loss

    @property
    def name(self):
        return "Metric Selection Model"



class PredictionIntervalModel(PredictiveModel):

    def __init__(self, input_size, y_size, learn_all_q, quantiles, miscoverage_level, hidden_size, num_layers, dropout, lr, wd, device, args):
        super(PredictionIntervalModel, self).__init__()
        self.device = device
        self.dropout = dropout
        self.input_size = input_size
        self.learn_all_q = learn_all_q
        nn_input_size = input_size + 1 if learn_all_q else input_size
        nn_output_size = y_size if learn_all_q else y_size * 2
        self.model = vanilla_nn(input_size=nn_input_size, output_size=nn_output_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                dropout=dropout).to(device)
        self.best_va_model = deepcopy(self.model)
        self.model.q_list = torch.Tensor([miscoverage_level/2]).to(device)
        self.optimizers = torch.optim.Adam(self.model.parameters(),
                                           lr=lr, weight_decay=wd)
        self.loss_fn = batch_qr_loss if learn_all_q else two_headed_qr_loss
        self.quantiles = quantiles

    def calc_train_loss(self, loss_fn, x, y, take_step, args, index):
        if self.learn_all_q:
            args.q_list = self.quantiles.to(self.device)  #torch.rand(30).to(self.device)#torch.rand(self.num_q).to(self.device)
        self.loss(loss_fn, x, y, take_step=take_step, args=args)

    def calc_va_loss(self, loss_fn, x, y, args, weights=None):
        if self.learn_all_q:
            args.q_list = self.quantiles.to(self.device)  # self.quantiles.to(self.device)  # torch.linspace(0.01, 0.99, 99).to(self.device)
        with torch.no_grad():
            return self.loss(loss_fn, x, y, take_step=False, args=args, weights=weights)

    def estimate_interval(self, x, alpha, use_best_va_model=True):
        if self.learn_all_q:
            upper_q_in = torch.cat([x, torch.ones(len(x), 1).to(x.device) * (1-alpha/2)], dim=-1)
            upper_q = self.predict(upper_q_in, use_best_va_model).squeeze()
            lower_q_in = torch.cat([x, torch.ones(len(x), 1).to(x.device) * (alpha/2)], dim=-1)
            lower_q = self.predict(lower_q_in, use_best_va_model).squeeze()
            return torch.cat([lower_q.unsqueeze(-1), upper_q.unsqueeze(-1)], dim=-1)
        else:
            return self.predict(x)

    @property
    def name(self):
        method_name = 'QR'
        return f"{method_name} with learn_all_q={self.learn_all_q}"

class UncertaintyByPredictionInterval(PredictiveModel, UncertaintyModel):

    def __init__(self, input_size, y_size, hidden_size, num_layers, q_list, dropout, lr, wd, device):
        super(UncertaintyByPredictionInterval, self).__init__()
        self.device = device
        self.dropout = dropout
        self.input_size = input_size
        self.q_list = q_list
        self.model = vanilla_nn(input_size=input_size, output_size=y_size * 2,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                dropout=dropout).to(device)
        self.model.q_list = q_list
        self.optimizers = torch.optim.Adam(self.model.parameters(),
                                           lr=lr, weight_decay=wd)
        self.loss_fn = two_headed_qr_loss

    def predict_uncertainty(self, x, use_best_va_model=True):
        pred = self.predict(x, use_best_va_model)
        return -(pred[:, 1] - pred[:, 0])

    @property
    def name(self):
        return "OQR"


class UncertaintyByResidual(UncertaintyModel):

    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout, lr, wd, device):
        super().__init__()
        self.mse_model = MSEModel(input_size, output_size, hidden_size, num_layers, dropout, lr, wd, device)

    def predict_uncertainty(self, x, use_best_va_model=True):
        pred = self.mse_model.predict(x, use_best_va_model)
        return -(pred.squeeze())

    def train(self, x_tr, y_tr, x_va, y_va, args, print_starting_to_train=True):
        if print_starting_to_train:
            print(f"starting to train: {self.name}")
        self.mse_model.train(x_tr, y_tr, x_va, y_va, args, print_starting_to_train=False)

    @property
    def name(self):
        return "Residual"


class SelectiveNet(PredictiveModel, UncertaintyModel):

    def __init__(self, input_size, y_size, hidden_size, num_layers, dropout, lr, wd, device, coverage=0.9):

        super().__init__()
        self.coverage = coverage
        self.model = selective_net_nn(input_size=input_size, y_size=y_size,
                                      hidden_size=hidden_size,
                                      num_layers=num_layers,
                                      dropout=dropout).to(device)
        self.device = device
        self.model.coverage = coverage
        self.loss_fn = selective_net_loss
        self.optimizers = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)

    def predict(self, x, use_best_va_model=True):
        return self.forward(x, use_best_va_model)[0]

    def predict_uncertainty(self, x, use_best_va_model=True):
        return self.forward(x, use_best_va_model)[1]

    def forward(self, x, use_best_va_model=True):

        with torch.no_grad():
            if use_best_va_model:
                pred = self.best_va_model(x)
            else:
                pred = self.model(x)

        return pred

    @property
    def name(self):
        return "SelectiveNet"


# class ClassificationModel(PredictiveModel):
#
#     def __init__(self, input_size, hidden_size, num_layers, dropout, lr, wd, device):
#         super().__init__()
#         self.model = binary_classification_nn(input_size=input_size,
#                                               hidden_size=hidden_size,
#                                               num_layers=num_layers,
#                                               dropout=dropout).to(device)
#         self.loss_fn = binary_classification_loss
#         self.optimizers = torch.optim.Adam(self.model.parameters(),
#                                            lr=lr, weight_decay=wd)


class MSEModel(PredictiveModel):

    def __init__(self,
                 input_size, output_size, hidden_size, num_layers, dropout, lr, wd, device):
        super().__init__()
        self.model = vanilla_nn(input_size=input_size, output_size=output_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                dropout=dropout).to(device)
        self.optimizers = torch.optim.Adam(self.model.parameters(),
                                           lr=lr, weight_decay=wd)
        self.loss_fn = batch_mse_loss
        self.device = device

    @property
    def name(self):
        return "MSE Model"


if __name__ == '__main__':
    pass
    # temp_model = QModelEns(input_size=1, output_size=1, hidden_size=10,
    #                        num_layers=2, lr=0.01, wd=0.0, num_ens=5,
    #                        device=torch.device('cuda:0'))
