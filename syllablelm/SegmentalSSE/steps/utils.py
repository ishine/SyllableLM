import math
import pickle
import math
import numpy as np
import torch
import torch.nn.functional as F
from .trainer_utils import AverageMeter

def find_boundary_matches(gt, pred, tolerance):
    """
    gt: list of ground truth boundaries
    pred: list of predicted boundaries
    all in seconds
    """
    gt_pointer = 0
    pred_pointer = 0
    gt_len = len(gt)
    pred_len = len(pred)
    match_pred = 0
    match_gt = 0
    while gt_pointer < gt_len and pred_pointer < pred_len:
        if np.abs(gt[gt_pointer] - pred[pred_pointer]) <= tolerance:
            match_gt += 1
            match_pred += 1
            gt_pointer += 1
            pred_pointer += 1
        elif gt[gt_pointer] > pred[pred_pointer]:
            pred_pointer += 1
        else:
            gt_pointer += 1
    # for pred_i in pred:
    #     min_dist = np.abs(gt - pred_i).min()
    #     match_pred += (min_dist <= tolerance)
    # for y_i in gt:
    #     min_dist = np.abs(pred - y_i).min()
    #     match_gt += (min_dist <= tolerance)
    return match_gt, match_pred, gt_len, pred_len


def Margin_InfoNCE_loss(S, margin, img_id=None):
    target = torch.LongTensor(list(range(S.size(0)))).to(S.device)
    deltas = margin * torch.eye(S.size(0)).to(S.device)
    S = S - deltas
    very_neg = torch.tensor(-10000.).to(S)
    if img_id is not None:
        img_id_equal_matrix = torch.from_numpy((img_id[:,None] == img_id[None,:])).to(S)
        diag_mask = (-1. * torch.eye(S.size(0)).to(S.device)) + torch.ones_like(S)
        mask = img_id_equal_matrix * diag_mask * very_neg # diag are all 0; for non-diag entries, -10000 if corresponding img_ids' are equal, otherwise 0
        # print(f"mask that is to be added to S: {mask}")
        # print(f"sum of entries of mask for each audio: {mask.sum(1)}") # actually -10000 rarely happened
        S = S + mask

    I2C_loss = F.nll_loss(F.log_softmax(S, dim=1), target, reduction='none')            
    C2I_loss = F.nll_loss(F.log_softmax(S.t(), dim=1), target, reduction='none')        
    loss = I2C_loss + C2I_loss
    return loss

def calc_recalls_from_S_one_to_many_coarse(S, row_img_id, column_img_id):
    # image is row, audio is colum
    row = S.size(0)
    column = S.size(1)
    I2A_scores, I2A_ind = S.topk(100, 1)
    A2I_scores, A2I_ind = S.topk(100, 0)
    A_r1 = AverageMeter()
    A_r5 = AverageMeter()
    A_r10 = AverageMeter()
    A_r100 = AverageMeter()
    I_r1 = AverageMeter()
    I_r5 = AverageMeter()
    I_r10 = AverageMeter()
    I_r100 = AverageMeter()
    for i in range(row):
        A_foundind = -1
        for ind in range(100):
            if row_img_id[i] == column_img_id[I2A_ind[i, ind]]:
                A_foundind = ind
                break
        # do r1s
        if A_foundind == 0:
            A_r1.update(1)
        else:
            A_r1.update(0)
        # do r5s
        if A_foundind >= 0 and A_foundind < 5:
            A_r5.update(1)
        else:
            A_r5.update(0)
        # do r10s
        if A_foundind >= 0 and A_foundind < 10:
            A_r10.update(1)
        else:
            A_r10.update(0)
        # do r10s
        if A_foundind >= 0 and A_foundind < 100:
            A_r100.update(1)
        else:
            A_r100.update(0)

    for i in range(column):
        I_foundind = -1
        for ind in range(100):
            if column_img_id[i] == row_img_id[A2I_ind[ind, i]]:
                I_foundind = ind
                break
        # do r1s
        if I_foundind == 0:
            I_r1.update(1)
        else:
            I_r1.update(0)
        # do r5s
        if I_foundind >= 0 and I_foundind < 5:
            I_r5.update(1)
        else:
            I_r5.update(0)
        # do r10s
        if I_foundind >= 0 and I_foundind < 10:
            I_r10.update(1)
        else:
            I_r10.update(0)
        # do r100s
        if I_foundind >= 0 and I_foundind < 100:
            I_r100.update(1)
        else:
            I_r100.update(0)

    recalls = {'A_r1':A_r1.avg, 'A_r5':A_r5.avg, 'A_r10':A_r10.avg, 'A_r100':A_r100.avg,
                'I_r1':I_r1.avg, 'I_r5':I_r5.avg, 'I_r10':I_r10.avg, 'I_r100':I_r100.avg}
                #'A_meanR':A_meanR.avg, 'I_meanR':I_meanR.avg}

    return recalls   