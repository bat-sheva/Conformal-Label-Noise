import abc
import time

import numpy as np
import torch
from helper import compute_coverages_and_avg_interval_len, pearsons_corr2d, HSIC
from typing import Union


class Loss(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def compute_loss(self, y, intervals):
        pass

    @abc.abstractmethod
    def min_second_derivative(self):
        pass

    @abc.abstractmethod
    def max_val(self):
        pass


class MiscoverageLoss(Loss):

    def __init__(self):
        super().__init__()

    def max_val(self):
        return 1

    def min_second_derivative(self):
        return -np.inf

    def compute_loss(self, y, intervals):
        if len(intervals.shape) == 1:
            intervals = intervals.unsqueeze(0).repeat(y.shape[0], 1)
        return ((y > intervals[..., 1]) | (y < intervals[..., 0])).float()


def get_hash(y, a, b):
    return str(id(y)) + str(id(a)) + str(id(b))


class SmoothedMiscoverageLoss(Loss):

    def __init__(self, c=1, d=1, p=1):
        super().__init__()
        self.c = c
        self.d = d
        self.p = p
        self.sigmoid_cache = {}
        self.sigmoid_tag_cache = {}

        self.scaling_cache = {}
        self.scaling_tag_cache = {}
        self.scaling_tag_aux_cache = {}

    def clear_cache(self):
        # self.clear_dict(self.sigmoid_cache)
        # self.clear_dict(self.sigmoid_tag_cache)
        # del self.sigmoid_cache
        # del self.sigmoid_tag_cache
        self.sigmoid_cache = {}
        self.sigmoid_tag_cache = {}
        self.clear_scaling_cache()
        # torch.cuda.empty_cache()

    def clear_scaling_cache(self):
        # self.clear_dict(self.scaling_cache)
        # self.clear_dict(self.scaling_tag_cache)
        # self.clear_dict(self.scaling_tag_aux_cache)
        # del self.scaling_cache
        # del self.scaling_tag_cache
        # del self.scaling_tag_aux_cache
        self.scaling_cache = {}
        self.scaling_tag_cache = {}
        self.scaling_tag_aux_cache = {}

    def max_val(self):
        return 1

    def get_extreme_second_derivative(self, intervals, is_min, range=None, step=0.001):
        if range is not None:
            return self.get_extreme_second_derivative_aux(intervals, is_min, range=range, step=step)
        range_1_q = self.get_extreme_second_derivative_aux(intervals, is_min, range=[0, 7.5], step=step)
        return range_1_q
        # range_2_q = self.get_extreme_second_derivative_aux(intervals, is_min, range=[7.5, 15], step=step)
        # if is_min:
        #     return torch.min(range_1_q, range_2_q)
        # else:
        #     return torch.max(range_1_q, range_2_q)

    def get_extreme_second_derivative_aux(self, intervals, is_min, range, step=0.001):
        if intervals is None:
            intervals = torch.Tensor([[-1, 1]])
        if len(intervals.shape) == 1:
            intervals = intervals.unsqueeze(0)

        y = torch.arange(range[0], range[1], step, device=intervals.device)
        y_size = y.shape[0]
        y_rep = y.unsqueeze(0).repeat(intervals.shape[0], 1).flatten(0, 1)
        intervals_rep = intervals.unsqueeze(1).repeat(1, y_size, 1).flatten(0, 1)
        y_rep = torch.cat([y_rep, intervals.mean(dim=1) + 0.0000001], dim=0)
        intervals_rep = torch.cat([intervals_rep, intervals], dim=0)
        # y_rep = self.scaling_func(y_rep, intervals_rep[..., 0], intervals_rep[..., 1])
        self.clear_cache()

        unflatten = torch.nn.Unflatten(0, (intervals.shape[0], y_size))
        result = self.smoothed_L_tag_tag(y_rep, intervals_rep)
        mean_interval_loss_tag_tag = result[-intervals.shape[0]:]
        result = result[:-intervals.shape[0]]
        smoothed_loss_tag_tag = unflatten(result)
        smoothed_loss_tag_tag = torch.cat([smoothed_loss_tag_tag, mean_interval_loss_tag_tag.unsqueeze(1)], dim=1)
        if is_min:
            smoothed_loss_tag_tag[smoothed_loss_tag_tag.isnan()] = np.inf
            return smoothed_loss_tag_tag.min(dim=1)[0].squeeze()
        else:
            smoothed_loss_tag_tag[smoothed_loss_tag_tag.isnan()] = -np.inf
            return smoothed_loss_tag_tag.max(dim=1)[0].squeeze()

    def min_second_derivative(self, intervals=None, range=None, step=0.0015):
        return self.get_extreme_second_derivative(intervals, is_min=True, range=range, step=step)

    def max_second_derivative(self, intervals=None, range=None, step=0.0015):
        return self.get_extreme_second_derivative(intervals, is_min=False, range=range, step=step)

    def compute_loss(self, y, intervals):
        if len(intervals.shape) == 1:
            intervals = intervals.unsqueeze(0).repeat(y.shape[0], 1)
        return self.compute_loss_aux(y, intervals) ** self.p

    def compute_loss_aux(self, y, intervals):
        a, b = intervals[..., 0], intervals[..., 1]
        return 2 * self.sigmoid(self.d * self.scaling_func(y, a, b)) - 1

    def get_intersection_point(self):
        return self.compute_loss(torch.Tensor([1]), torch.Tensor([[-1, 1]])).item()

    def sigmoid(self, x):
        if x in self.sigmoid_cache:
            return self.sigmoid_cache[x]
        res = 1 / (1 + torch.exp(-x))
        self.sigmoid_cache[x] = res
        return res

    def sigmoid_tag(self, x):
        if x in self.sigmoid_tag_cache:
            return self.sigmoid_tag_cache[x]
        res = self.sigmoid(x) * (1 - self.sigmoid(x))
        self.sigmoid_tag_cache[x] = res
        return res

    def sigmoid_tag_tag(self, x):
        # return self.sigmoid_tag(x) * (1 - self.sigmoid(x)) + self.sigmoid(x) * (-self.sigmoid_tag(x))
        # return self.sigmoid_tag(x) - self.sigmoid_tag(x)*self.sigmoid(x) - self.sigmoid(x) * self.sigmoid_tag(x)
        return self.sigmoid_tag(x) - 2 * self.sigmoid(x) * self.sigmoid_tag(x)

    def a_plus_b_minus_2_y(self, y, a, b):
        if get_hash(y, a, b) in self.scaling_tag_aux_cache:
            return self.scaling_tag_aux_cache[get_hash(y, a, b)]
        res = (a + b - 2 * y)
        self.scaling_tag_aux_cache[get_hash(y, a, b)] = res
        return res

    def scaling_func(self, y, a, b):
        if get_hash(y, a, b) in self.scaling_cache:
            return self.scaling_cache[get_hash(y, a, b)]
        a_plus_b_minus_2_y = self.a_plus_b_minus_2_y(y, a, b)
        res = ((a_plus_b_minus_2_y / (b - a)) ** 2) ** self.c
        self.scaling_cache[get_hash(y, a, b)] = res
        return res

    def scaling_func_tag(self, y, interval):
        a, b = interval[..., 0], interval[..., 1]
        if get_hash(y, a, b) in self.scaling_tag_cache:
            return self.scaling_tag_cache[get_hash(y, a, b)]
        top = (4 * self.c) * self.scaling_func(y, a, b)
        bot = self.a_plus_b_minus_2_y(y, a, b)
        res = -top / bot
        self.scaling_tag_cache[get_hash(y, a, b)] = res
        return res

    def scaling_func_tag_tag(self, y, interval):
        a, b = interval[..., 0], interval[..., 1]
        top = (2 * (2 * self.c - 1)) * self.scaling_func_tag(y, interval)
        bot = self.a_plus_b_minus_2_y(y, a, b)
        return -top / bot

    def smoothed_L_tag_aux(self, y, interval):
        a, b = interval[..., 0], interval[..., 1]
        return 2 * self.sigmoid_tag(self.d * self.scaling_func(y, a, b)) * (
                self.d * self.scaling_func_tag(y, interval))

    def smoothed_L_tag_tag_aux(self, y, interval):
        a, b = interval[..., 0], interval[..., 1]
        y_scaled = self.d * self.scaling_func(y, a, b)
        scaling_func_tag = self.d * self.scaling_func_tag(y, interval)
        scaling_func_tag_tag = self.d * self.scaling_func_tag_tag(y, interval)
        self.clear_scaling_cache()
        sigmoid_tag_tag = self.sigmoid_tag_tag(y_scaled)
        sigmoid_tag = self.sigmoid_tag(y_scaled)
        self.clear_cache()
        res = 2 * (sigmoid_tag_tag * (scaling_func_tag ** 2) + sigmoid_tag * scaling_func_tag_tag)
        return res

    def smoothed_L_tag(self, y, interval):
        if len(interval.shape) == 1:
            interval = interval.unsqueeze(0).repeat(y.shape[0], 1)
        p = self.p
        if p == 1:
            return self.smoothed_L_tag_aux(y, interval)
        l = self.compute_loss_aux(y, interval)
        l_tag = self.smoothed_L_tag_aux(y, interval)
        return p * (l ** (p - 1)) * l_tag

    def smoothed_L_tag_tag(self, y, interval):
        if len(interval.shape) == 1:
            interval = interval.unsqueeze(0).repeat(y.shape[0], 1)
        p = self.p
        if p == 1:
            res = self.smoothed_L_tag_tag_aux(y, interval)
            self.clear_cache()
            return res
        self.clear_cache()
        l = self.compute_loss_aux(y, interval)
        l_tag = self.smoothed_L_tag_aux(y, interval)
        self.clear_cache()
        l_tag_tag = self.smoothed_L_tag_tag_aux(y, interval)
        res = p * (p - 1) * (l ** (p - 2)) * (l_tag ** 2) + \
              p * (l ** (p - 1)) * l_tag_tag
        return res


def independence_penalty(y, pred_l, pred_u, pearsons_corr_multiplier=1, hsic_multiplier=0, y_multiplier=100):
    """
    Computes the independence penalty given the true label and the prediced upper and lower quantiles.
    Parameters
    ----------
    y - the true label of a feature vector.
    pred_l - the predicted lower bound
    pred_u - the prediced upper bound
    pearsons_corr_multiplier - multiplier of R_corr
    hsic_multiplier - multiplier of R_HSIC
    y_multiplier - multiplier of y for numeric stability

    Returns
    -------
    The independence penalty R
    """

    if pearsons_corr_multiplier == 0 and hsic_multiplier == 0:
        return 0

    is_in_interval, interval_sizes = compute_coverages_and_avg_interval_len(y.view(-1) * y_multiplier,
                                                                            pred_l * y_multiplier,
                                                                            pred_u * y_multiplier)
    partial_interval_sizes = interval_sizes[abs(torch.min(is_in_interval, dim=1)[0] -
                                                torch.max(is_in_interval, dim=1)[0]) > 0.05, :]
    partial_is_in_interval = is_in_interval[abs(torch.min(is_in_interval, dim=1)[0] -
                                                torch.max(is_in_interval, dim=1)[0]) > 0.05, :]

    if partial_interval_sizes.shape[0] > 0 and pearsons_corr_multiplier != 0:
        corrs = pearsons_corr2d(partial_interval_sizes, partial_is_in_interval)
        pearsons_corr_loss = torch.mean((torch.abs(corrs)))
        if pearsons_corr_loss.isnan().item():
            pearsons_corr_loss = 0
    else:
        pearsons_corr_loss = 0

    hsic_loss = 0
    if partial_interval_sizes.shape[0] > 0 and hsic_multiplier != 0:
        n = partial_is_in_interval.shape[1]
        data_size_for_hsic = 512
        for i in range(partial_is_in_interval.shape[0]):

            v = partial_is_in_interval[i, :].reshape((n, 1))
            l = partial_interval_sizes[i, :].reshape((n, 1))
            v = v[:data_size_for_hsic]
            l = l[:data_size_for_hsic]
            if torch.max(v) - torch.min(v) > 0.05:  # in order to not get hsic = 0
                curr_hsic = torch.abs(torch.sqrt(HSIC(v, l)))
            else:
                curr_hsic = 0

            hsic_loss += curr_hsic
        hsic_loss = hsic_loss / partial_interval_sizes.shape[0]

    penalty = pearsons_corr_loss * pearsons_corr_multiplier + hsic_loss * hsic_multiplier

    return penalty


class MetricLoss:
    def __init__(self, calc_metric, calc_calibrated_pred, desired_level, coeff, x_ca, y_ca, cal_pred, miscoverage_level,
                 train_preds, val_preds):
        self.calc_metric = calc_metric
        self.calc_calibrated_pred = calc_calibrated_pred
        self.desired_level = desired_level
        self.coeff = coeff
        self.x_ca = x_ca
        self.y_ca = y_ca
        self.cal_pred = cal_pred
        self.miscoverage_level = miscoverage_level
        self.train_preds = train_preds
        self.val_preds = val_preds

    def loss(self, scoring_model, y, x, device, args, weights=None, index=None, is_val=False):
        calibration_scores = scoring_model(self.x_ca).detach()
        scores = scoring_model(x).squeeze()
        regularization = (scores - 1).square().mean()
        preds = self.val_preds if is_val else self.train_preds[index]
        calibrated_preds = self.calc_calibrated_pred(preds, self.miscoverage_level, calibration_scores,
                                                     self.y_ca, self.cal_pred).detach()
        metric_level = self.calc_metric(calibrated_preds, y, scores)
        if metric_level > self.desired_level:
            metric_loss = metric_level
        else:
            metric_loss = 0
        return metric_loss + self.coeff * regularization


class MetricCalibrationLoss:
    def __init__(self, calc_metric, calc_calibrated_pred_for_metric, desired_level, coeff, x_ca, y_ca, cal_pred,
                 miscoverage_level, train_preds, val_preds):
        self.calc_metric = calc_metric
        self.calc_calibrated_pred_for_metric = calc_calibrated_pred_for_metric
        self.desired_level = desired_level
        self.coeff = coeff
        self.x_ca = x_ca
        self.y_ca = y_ca
        self.cal_pred = cal_pred
        self.miscoverage_level = miscoverage_level
        self.train_preds = train_preds
        self.val_preds = val_preds

    def loss(self, scoring_model, y, x, device, args, weights=None, index=None, is_val=False):
        cal_scores = scoring_model(self.x_ca).squeeze()
        scores = scoring_model(x).squeeze()
        intervals = self.val_preds if is_val else self.train_preds[index]
        calibrated_intervals = self.calc_calibrated_pred_for_metric(scores, intervals, self.miscoverage_level,
                                                                    self.y_ca, self.cal_pred, cal_scores)
        regularization = interval_pinball_loss(y, calibrated_intervals[:, 1], calibrated_intervals[:, 0],
                                               self.miscoverage_level)

        metric_level = self.calc_metric(calibrated_intervals, y, scores)
        if metric_level > self.desired_level:
            metric_loss = metric_level
        else:
            metric_loss = 0
        return metric_loss + self.coeff * regularization


def pinball_loss(y, q_hat, q_level, apply_mean=True) -> Union[float, np.ndarray]:
    diff = q_hat - y
    mask = (diff.ge(0).float() - q_level).detach()  # / q_rep

    loss = (mask * diff)
    if apply_mean:
        return loss.mean()
    else:
        return loss


def interval_pinball_loss(y, upper_q, lower_q, miscoverage_level, apply_mean=True):
    return (pinball_loss(y, upper_q, 1 - miscoverage_level / 2, apply_mean) + pinball_loss(y, lower_q,
                                                                                           miscoverage_level / 2,
                                                                                           apply_mean)) / 2


def two_headed_qr_loss(model, y, x, device, args, weights=None, index=None, is_val=False):
    q_list = model.q_list
    alpha_low = min(q_list[0].item(), 1 - q_list[0].item())
    alpha_high = 1 - alpha_low
    pred = model(x).squeeze()

    pred_u = pred[:, 1]
    pred_l = pred[:, 0]

    y_rep = torch.cat([y, y], dim=0).squeeze()
    q_hat = torch.cat([pred_l, pred_u], dim=0).squeeze()
    q_level = torch.zeros_like(y_rep)
    q_level[:len(q_level) // 2] = alpha_low
    q_level[len(q_level) // 2:] = alpha_high
    diff = q_hat - y_rep
    mask = (diff.ge(0).float() - q_level).detach()  # / q_rep

    pb_loss = (mask * diff).mean()
    # pb_loss = (pinball_loss(y.squeeze(), pred_u, alpha_high) + pinball_loss(y.squeeze(), pred_l, alpha_low)) / 2

    pearsons_corr_multiplier = 0
    hsic_multiplier = 0
    independence_loss = independence_penalty(y, pred_l.unsqueeze(0), pred_u.unsqueeze(0), pearsons_corr_multiplier,
                                             hsic_multiplier)

    loss = pb_loss + independence_loss

    return loss


def selective_net_loss(model, y, x, device, args, weights=None):
    alpha = 0.5
    c = model.coverage
    lambda_ = 32
    f, g, h = model(x)
    l = (y.squeeze() - f) ** 2
    coverage = g.mean()
    r_l = ((1 / l.shape[0]) * l @ g) / coverage
    phi = torch.relu(c - coverage) ** 2
    f_g_loss = r_l + lambda_ * phi
    h_loss = ((h - y) ** 2).mean()

    return alpha * f_g_loss + (1 - alpha) * h_loss


def binary_classification_loss(model, y, x, device, args, weights=None):
    pred = model(x)
    return (-y * torch.log(pred) - (1 - y) * torch.log(1 - pred)).mean()


def batch_mse_loss(model, y, x, device, args, weights=None):
    pred = model(x).squeeze()
    loss = ((pred - y.squeeze()) ** 2).mean()
    return loss


def batch_qr_loss(model, y, x, device, args, weights=None, index=None, is_val=False):
    num_pts = y.size(0)
    q_list = args.q_list
    with torch.no_grad():
        l_list = q_list[q_list <= 0.5]
        u_list = 1.0 - l_list

    q_list = torch.cat([l_list, u_list], dim=0)
    num_q = q_list.shape[0]

    l_rep = l_list.view(-1, 1).repeat(1, num_pts).view(-1, 1)
    num_l = l_rep.size(0)

    q_rep = q_list.unsqueeze(0).repeat(num_pts, 1).to(device).flatten()
    y_stacked = y.repeat(1, num_q).flatten()

    if x is None:
        model_in = q_list
    else:
        x_stacked = x.unsqueeze(1).repeat(1, num_q, 1).flatten(0, 1)
        model_in = torch.cat([x_stacked, q_rep.unsqueeze(-1)], dim=-1)

    pred_y = model(model_in).squeeze()

    diff = pred_y - y_stacked
    mask = (diff.ge(0).float() - q_rep).detach()  # / q_rep

    pinball_losses = (mask * diff)
    if weights is not None:
        pinball_loss = pinball_losses.squeeze() @ weights.repeat(num_q) / (num_q * weights.sum())
    else:
        pinball_loss = pinball_losses.mean()

    pearsons_corr_multiplier = 0
    hsic_multiplier = 0
    pred_l = pred_y[:num_l].view(num_q // 2, num_pts)
    pred_u = pred_y[num_l:].view(num_q // 2, num_pts)
    independence_loss = independence_penalty(y, pred_l, pred_u, pearsons_corr_multiplier, hsic_multiplier)
    # independence_loss = 0
    loss = pinball_loss + independence_loss

    return loss


def batch_interval_loss(model, y, x, q_list, device, args, weights=None):
    """
    implementation of interval score, for batch of quantiles
    """
    num_pts = y.size(0)
    num_q = q_list.size(0)

    with torch.no_grad():
        l_list = torch.min(torch.stack([q_list, 1 - q_list], dim=1), dim=1)[0].to(device)
        u_list = 1.0 - l_list

    l_rep = l_list.view(-1, 1).repeat(1, num_pts).view(-1, 1)
    u_rep = u_list.view(-1, 1).repeat(1, num_pts).view(-1, 1)
    num_l = l_rep.size(0)

    if x is None:
        model_in = torch.cat([l_list, u_list], dim=0)
    else:
        x_stacked = x.repeat(num_q, 1)
        l_in = torch.cat([x_stacked, l_rep], dim=1)
        u_in = torch.cat([x_stacked, u_rep], dim=1)
        model_in = torch.cat([l_in, u_in], dim=0)

    pred_y = model(model_in)
    pred_l = pred_y[:num_l].view(num_q, num_pts)
    pred_u = pred_y[num_l:].view(num_q, num_pts)

    below_l = (pred_l - y.view(-1)).gt(0)
    above_u = (y.view(-1) - pred_u).gt(0)

    interval_score_losses = (pred_u - pred_l) + \
                            (1.0 / l_list).view(-1, 1).to(device) * (pred_l - y.view(-1)) * below_l + \
                            (1.0 / l_list).view(-1, 1).to(device) * (y.view(-1) - pred_u) * above_u

    if weights is not None:
        interval_score_loss = interval_score_losses @ weights
    else:
        interval_score_loss = interval_score_losses.mean()

    pearsons_corr_multiplier = 0
    hsic_multiplier = 0
    independence_loss = independence_penalty(y, pred_l, pred_u, pearsons_corr_multiplier, hsic_multiplier)

    loss = interval_score_loss + independence_loss

    return loss
