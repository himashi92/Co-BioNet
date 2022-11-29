import torch
import torch.nn as nn
from torch.autograd import Variable


def Binary_dice_loss(predictive, target, ep=1e-8):
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss


def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def kl_loss(inputs, targets, ep=1e-8):
    kl_loss = nn.KLDivLoss(reduction='mean')
    consist_loss = kl_loss(torch.log(inputs + ep), targets)
    return consist_loss


def soft_ce_loss(inputs, target, ep=1e-8):
    logprobs = torch.log(inputs + ep)
    return torch.mean(-(target[:, 0, ...] * logprobs[:, 0, ...] + target[:, 1, ...] * logprobs[:, 1, ...]))


def mse_loss(input1, input2):
    return torch.mean((input1 - input2) ** 2)


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersection = torch.sum(score * target)
        union = torch.sum(score * score) + torch.sum(target * target) + smooth
        loss = 1 - intersection / union
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


CE = torch.nn.BCELoss()
mse = torch.nn.MSELoss()


def loss_diff1(u_prediction_1, u_prediction_2):
    loss_a = 0.0

    for i in range(u_prediction_2.size(1)):
        loss_a = CE(u_prediction_1[:, i, ...].clamp(1e-8, 1 - 1e-7),
                                 Variable(u_prediction_2[:, i, ...].float(), requires_grad=False))

    loss_diff_avg = loss_a.mean().item()
    return loss_diff_avg


def loss_diff2(u_prediction_1, u_prediction_2):
    loss_b = 0.0

    for i in range(u_prediction_2.size(1)):
        loss_b = CE(u_prediction_2[:, i, ...].clamp(1e-8, 1 - 1e-7),
                                 Variable(u_prediction_1[:, i, ...], requires_grad=False))

    loss_diff_avg = loss_b.mean().item()
    return loss_diff_avg


def loss_mask(u_prediction_1, u_prediction_2, critic_segs, T_m):
    loss_a = 0.0
    for i in range(u_prediction_2.size(1)):
        gen_mask = (critic_segs[:, i, ...].squeeze(0) > T_m).float()
        loss_a = loss_a + gen_mask * CE(u_prediction_1[:, i, ...],
                                            Variable(u_prediction_2[:, i, ...].float(), requires_grad=False))

    loss_diff_avg = loss_a.mean() / u_prediction_2.size(1)

    return loss_diff_avg


def disc_loss(pred, target):
    real_loss1 = CE(target, torch.ones_like(target).float())
    fake_loss1 = CE(pred, torch.zeros_like(pred).float())

    loss = (1/2) * (real_loss1 + fake_loss1)

    return loss


def gen_loss(pred):
    fake_loss1 = CE(pred, torch.ones_like(pred).float())

    loss = fake_loss1

    return loss